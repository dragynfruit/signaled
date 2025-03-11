mod tee_source;

use futures_util::StreamExt;
use ratatui::text::Line;
use ratatui::widgets::BarChart;
use rodio::{Decoder, OutputStream, Source};
use std::cmp;
use std::collections::VecDeque;
use std::error::Error;
use std::io::{Read, Seek, SeekFrom, BufWriter};
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use log::{debug, error, info, trace, warn};

use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    event::{poll, read, Event, KeyCode, KeyEventKind, KeyModifiers},
};

use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    widgets::{Block, Borders, Gauge, Paragraph},
    Terminal,
};
use tui_logger::{ExtLogRecord, LogFormatter, TuiLoggerWidget};

use crate::tee_source::TeeSource;
use argh::FromArgs;
use rustfft::{num_complex::Complex, FftPlanner};

const STREAM_URL: &str = "https://radio.vern.cc/lofi.ogg";

#[derive(FromArgs)]
/// Radio playback program.
struct Args {
    /// stream url to play
    #[argh(option, short = 'u')]
    stream_url: Option<String>,
    /// initial volume (default 1.0)
    #[argh(option, short = 'v')]
    volume: Option<f32>,
    /// target frame duration in milliseconds (default 50)
    #[argh(option, short = 'f')]
    frame_duration: Option<u64>,
}

struct CustomFormatter {}

impl LogFormatter for CustomFormatter {
    fn min_width(&self) -> u16 {
        4
    }
    fn format(&self, _width: usize, evt: &ExtLogRecord) -> Vec<Line> {
        vec![Line::from(format!("{}: {}", evt.level, evt.msg()))]
    }
}

struct ChannelReader {
    receiver: Receiver<Vec<u8>>,
    data: Vec<u8>,
    pos: usize,
}

impl ChannelReader {
    fn new(receiver: Receiver<Vec<u8>>) -> Self {
        info!("Initializing ChannelReader");
        Self {
            receiver,
            data: Vec::new(),
            pos: 0,
        }
    }
}

impl Read for ChannelReader {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        while self.pos == self.data.len() {
            match self.receiver.recv() {
                Ok(chunk) => {
                    debug!("Received {} bytes of data", chunk.len());
                    self.data.extend(chunk);
                }
                Err(e) => {
                    error!("Receiver error: {}", e);
                    return Ok(0);
                }
            }
        }
        let available = self.data.len() - self.pos;
        let n = cmp::min(out.len(), available);
        out[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        trace!("Reading {} bytes", n);
        self.pos += n;
        if self.pos >= 16384 {
            let drain_amount = self.pos / 2;
            self.data.drain(0..drain_amount);
            self.pos -= drain_amount;
        }
        Ok(n)
    }
}

impl Seek for ChannelReader {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let new_pos: i64 = match pos {
            SeekFrom::Start(offset) => offset as i64,
            SeekFrom::Current(offset) => self.pos as i64 + offset,
            SeekFrom::End(offset) => self.data.len() as i64 + offset,
        };
        if new_pos < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid seek position",
            ));
        }
        self.pos = new_pos as usize;
        Ok(self.pos as u64)
    }
}

unsafe impl Sync for ChannelReader {}

struct SharedChannelReader(Arc<Mutex<ChannelReader>>);

impl Read for SharedChannelReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().read(buf)
    }
}

impl Seek for SharedChannelReader {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.0.lock().unwrap().seek(pos)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args: Args = argh::from_env();
    let stream_source = args.stream_url.unwrap_or_else(|| STREAM_URL.to_string());
    let init_volume = args.volume.unwrap_or(1.0);
    let target_millis = args.frame_duration.unwrap_or(50);

    tui_logger::init_logger(log::LevelFilter::Trace).unwrap();
    tui_logger::set_default_level(log::LevelFilter::Trace);
    info!("Stream source: {}", stream_source);
    info!("Starting radio playback program");

    let stream_display = stream_source.clone();

    let (_stream, stream_handle) = OutputStream::try_default()?;
    info!("Audio output initialized");
    let sink = rodio::Sink::try_new(&stream_handle)?;
    sink.set_volume(init_volume);
    info!("Sink created");
    let sink = Arc::new(sink);

    let (tx, rx) = channel::<Vec<u8>>();
    info!("Channel for audio chunks created");

    use std::sync::atomic::{AtomicUsize, Ordering};
    let packets_counter = Arc::new(AtomicUsize::new(0));
    let bytes_counter = Arc::new(AtomicUsize::new(0));

    {
        let packets_counter = packets_counter.clone();
        let bytes_counter = bytes_counter.clone();
        tokio::spawn(async move {
            info!("Starting stream download from {}", stream_source);
            let response = reqwest::Client::new().get(&stream_source).send().await;
            match response {
                Ok(resp) => {
                    info!("HTTP request successful");
                    let stream = resp.bytes_stream();
                    stream
                        .for_each(|chunk| async {
                            match chunk {
                                Ok(data) => {
                                    debug!("Downloaded chunk of {} bytes", data.len());
                                    packets_counter.fetch_add(1, Ordering::Relaxed);
                                    bytes_counter.fetch_add(data.len(), Ordering::Relaxed);
                                    let _ = tx.send(data.to_vec());
                                }
                                Err(e) => {
                                    warn!("Error downloading chunk: {}", e);
                                }
                            }
                        })
                        .await;
                }
                Err(e) => {
                    error!("HTTP request failed: {}", e);
                }
            }
        });
    }

    let inner_reader = Arc::new(Mutex::new(ChannelReader::new(rx)));

    let freq_buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));

    let packets_ui = packets_counter.clone();
    let bytes_ui = bytes_counter.clone();
    let inner_reader_ui = inner_reader.clone();
    let freq_buffer_ui = freq_buffer.clone();
    let sink_ui = sink.clone();
    let ui_handle = tokio::spawn(async move {
        enable_raw_mode().expect("Failed to enable raw mode");
        let mut buffered = BufWriter::new(std::io::stdout());
        execute!(buffered, EnterAlternateScreen).unwrap();
        let backend = CrosstermBackend::new(buffered);
        let mut terminal = Terminal::new(backend).unwrap();
        let target_frame_duration = Duration::from_millis(target_millis);

        let mut last_rate_instant = Instant::now();
        let mut packets_sec = 0;
        let mut bytes_sec = 0;

        let mut packets_samples: VecDeque<usize> = VecDeque::new();
        let mut bytes_samples: VecDeque<usize> = VecDeque::new();

        let mut volume: f32 = init_volume;
        let mut muted: bool = false;
        let mut previous_volume: f32 = init_volume;
        let mut fft_planner = FftPlanner::<f32>::new();

        // Create frequency labels once
        let mut freq_labels: Vec<&'static str> = Vec::with_capacity(512);
        for i in 0..512 {
            freq_labels.push(Box::leak(format!("{}Hz", i * 43).into_boxed_str()));
        }

        loop {
            let frame_start = Instant::now();

            if last_rate_instant.elapsed() >= Duration::from_secs(1) {
                let packets_instant = packets_ui.load(Ordering::Relaxed);
                let bytes_instant = bytes_ui.load(Ordering::Relaxed);
                packets_ui.store(0, Ordering::Relaxed);
                bytes_ui.store(0, Ordering::Relaxed);

                packets_samples.push_back(packets_instant);
                bytes_samples.push_back(bytes_instant);
                if packets_samples.len() > 5 {
                    packets_samples.pop_front();
                }
                if bytes_samples.len() > 5 {
                    bytes_samples.pop_front();
                }
                let packets_avg = packets_samples.iter().sum::<usize>() / packets_samples.len();
                let bytes_avg = bytes_samples.iter().sum::<usize>() / bytes_samples.len();
                packets_sec = packets_avg;
                bytes_sec = bytes_avg;
                last_rate_instant = Instant::now();
            }

            let (pos, total) = {
                let r = inner_reader_ui.lock().unwrap();
                (r.pos, r.data.len())
            };
            let progress = if total > 0 {
                pos as f64 / total as f64
            } else {
                0.0
            };

            terminal.draw(|f| {
                let size = f.area();
                let outer_chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints(
                        [
                            Constraint::Length(3),
                            Constraint::Min(10),
                            Constraint::Length(3),
                        ]
                        .as_ref(),
                    )
                    .split(size);

                let top_chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints(
                        [Constraint::Percentage(50), Constraint::Percentage(50)].as_ref(),
                    )
                    .split(outer_chunks[0]);

                let left_info_chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints(
                        [Constraint::Percentage(50), Constraint::Percentage(50)].as_ref(),
                    )
                    .split(top_chunks[0]);

                let url_paragraph = Paragraph::new(&*stream_display)
                    .block(Block::default().borders(Borders::ALL).title("Stream URL"));
                f.render_widget(url_paragraph, left_info_chunks[0]);

                let volume_percent = (volume * 50.0) as u16;
                let volume_gauge = Gauge::default()
                    .block(Block::default().borders(Borders::ALL).title("Volume"))
                    .gauge_style(Style::default().fg(Color::Blue))
                    .percent(volume_percent)
                    .label(format!("{:.0}%", volume_percent * 2));
                f.render_widget(volume_gauge, left_info_chunks[1]);

                let stat_chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints(
                        [Constraint::Percentage(50), Constraint::Percentage(50)].as_ref(),
                    )
                    .split(top_chunks[1]);

                let packets_paragraph = Paragraph::new(format!("Packets/sec: {}", packets_sec))
                    .block(Block::default().borders(Borders::ALL).title("Stats"));
                f.render_widget(packets_paragraph, stat_chunks[0]);

                let bytes_paragraph = Paragraph::new(format!("Bytes/sec: {}", bytes_sec))
                    .block(Block::default().borders(Borders::ALL).title("Stats"));
                f.render_widget(bytes_paragraph, stat_chunks[1]);

                let mid_chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints(
                        [Constraint::Percentage(33), Constraint::Percentage(67)].as_ref(),
                    )
                    .split(outer_chunks[1]);

                let log_widget = TuiLoggerWidget::default()
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title("Logs")
                            .border_style(Style::default().fg(Color::White)),
                    )
                    .style(Style::default().fg(Color::Cyan))
                    .opt_formatter(Some(Box::new(CustomFormatter {})));
                f.render_widget(log_widget, mid_chunks[0]);

                let gauge_value = (progress * 100.0) as u16;

                // Simplified frequency data computation without caching:
                let freq_data: Vec<(&'static str, u64)> = {
                    let samples_lock = freq_buffer_ui.lock().unwrap();
                    let n = 1024;
                    let len = samples_lock.len().min(n);
                    if len == 0 {
                        Vec::new()
                    } else {
                        // Copy data while under lock
                        let data_slice: Vec<f32> = samples_lock[samples_lock.len().saturating_sub(len)..].to_vec();
                        drop(samples_lock); // Release lock before CPU-intensive work
                        
                        let fft_input: Vec<Complex<f32>> = data_slice
                            .iter()
                            .map(|&s| Complex { re: s, im: 0.0 })
                            .collect();
                        let fft = fft_planner.plan_fft_forward(len);
                        let mut fft_out = fft_input;
                        fft.process(&mut fft_out);
                        
                        let scale = 44100 / len;
                        fft_out
                            .iter()
                            .take(len / 2)
                            .enumerate()
                            .map(|(i, c)| {
                                // Use pre-allocated frequency labels when possible
                                let label = if i < freq_labels.len() {
                                    freq_labels[i]
                                } else {
                                    &*Box::leak(format!("{}Hz", i * scale).into_boxed_str())
                                };
                                (label, c.norm() as u64)
                            })
                            .collect()
                    }
                };

                let barchart = BarChart::default()
                    .block(Block::default().borders(Borders::ALL).title("Visualizer"))
                    .data(&freq_data)
                    .bar_width(5)
                    .max(300)
                    .bar_style(Style::default().fg(Color::Green))
                    .value_style(Style::default().fg(Color::Yellow));
                f.render_widget(barchart, mid_chunks[1]);

                let gauge = Gauge::default()
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title("Buffer Progress"),
                    )
                    .gauge_style(Style::default().fg(Color::Magenta))
                    .percent(gauge_value)
                    .label(format!("{} / {} bytes", pos, total));
                f.render_widget(gauge, outer_chunks[2]);
            }).unwrap();

            if poll(Duration::from_millis(0)).unwrap() {
                if let Event::Key(key_event) = read().unwrap() {
                    if key_event.kind != KeyEventKind::Press {
                        continue;
                    }
                    match key_event.code {
                        KeyCode::Char('c')
                            if key_event.modifiers.contains(KeyModifiers::CONTROL) =>
                        {
                            info!("CTRL-C detected via UI input; stopping playback.");
                            sink_ui.stop();
                            disable_raw_mode().expect("Failed to disable raw mode");
                            execute!(terminal.backend_mut(), LeaveAlternateScreen).unwrap();
                            std::process::exit(0);
                        }
                        KeyCode::Char('m') => {
                            if !muted {
                                previous_volume = volume;
                                volume = 0.0;
                                muted = true;
                            } else {
                                volume = previous_volume;
                                muted = false;
                            }
                            sink_ui.set_volume(volume);
                            info!("Mute toggled. Current volume: {}", volume);
                        }
                        KeyCode::Up => {
                            volume = f32::min(volume + 0.1, 2.0);
                            sink_ui.set_volume(volume);
                            info!("Volume increased to {}", volume);
                        }
                        KeyCode::Down => {
                            volume = f32::max(volume - 0.1, 0.0);
                            sink_ui.set_volume(volume);
                            info!("Volume decreased to {}", volume);
                        }
                        _ => {}
                    }
                }
            }

            if pos >= total && total > 0 {
                tokio::time::sleep(Duration::from_secs(1)).await;
                let (pos_new, total_new) = {
                    let r = inner_reader_ui.lock().unwrap();
                    (r.pos, r.data.len())
                };
                if pos_new == pos && total_new == total {
                    break;
                }
            }

            let elapsed = frame_start.elapsed();
            if elapsed < target_frame_duration {
                tokio::time::sleep(target_frame_duration - elapsed).await;
            }
        }
        disable_raw_mode().expect("Failed to disable raw mode");
        execute!(terminal.backend_mut(), LeaveAlternateScreen).unwrap();
    });

    tokio::time::sleep(Duration::from_millis(200)).await;

    let shared_reader = SharedChannelReader(inner_reader);
    let decoder = Decoder::new(shared_reader)?.convert_samples::<f32>();
    info!("Decoder created; wrapping in TeeSource");
    let tee = TeeSource::new(decoder, freq_buffer);

    sink.append(tee);

    ui_handle.await?;
    info!("Playback finished; exiting program");
    Ok(())
}
