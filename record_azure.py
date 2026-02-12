#!/usr/bin/env python3

import os
import sys
import time
import argparse
from typing import Optional
from datetime import datetime

import cv2
import threading
from queue import Queue, Full, Empty

try:
    # Support both new and old pyk4a APIs
    from pyk4a import PyK4A, Config, ColorResolution
    try:
        from pyk4a import ColorFormat  # newer API
    except Exception:
        ColorFormat = None
    try:
        from pyk4a import ImageFormat  # older API
    except Exception:
        ImageFormat = None
except Exception as import_error:  # pragma: no cover
    PyK4A = None
    _IMPORT_ERROR = import_error
else:
    _IMPORT_ERROR = None


def ensure_output_dir(directory_path: str) -> None:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)


def get_next_exp_index(output_dir: str) -> int:
    # Find next exp_N.mp4 index in the directory
    existing = []
    try:
        for name in os.listdir(output_dir):
            if not name.lower().endswith(".mp4"):
                continue
            if name.startswith("exp_"):
                base = name[:-4]  # strip .mp4
                try:
                    n = int(base.split("exp_")[-1])
                    existing.append(n)
                except Exception:
                    continue
    except FileNotFoundError:
        ensure_output_dir(output_dir)

    next_idx = (max(existing) + 1) if existing else 1
    return next_idx


def build_output_path(output_dir: str, prefix: str = "exp_") -> str:
    next_idx = get_next_exp_index(output_dir)
    filename = f"exp_{next_idx}.mp4"
    return os.path.join(output_dir, filename)


def draw_rec_indicator(
    frame,
    is_recording: bool,
    elapsed_seconds: Optional[float] = None,
    episode_name: Optional[str] = None,
) -> None:
    if not is_recording:
        return
    h, w = frame.shape[:2]
    # Red circle and REC text
    cv2.circle(frame, (20, 20), 8, (0, 0, 255), thickness=-1)
    cv2.putText(
        frame,
        "REC",
        (40, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    if elapsed_seconds is not None:
        hrs = int(elapsed_seconds // 3600)
        mins = int((elapsed_seconds % 3600) // 60)
        secs = int(elapsed_seconds % 60)
        time_text = f"{hrs:02d}:{mins:02d}:{secs:02d}"
        cv2.putText(
            frame,
            time_text,
            (110, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if episode_name:
        cv2.putText(
            frame,
            episode_name,
            (220, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def draw_status_overlay(
    frame,
    is_recording: bool,
    output_dir: str,
    color_resolution: str,
    fps: float,
    next_index: int,
    current_filename: Optional[str],
) -> None:
    h, w = frame.shape[:2]
    y = 50
    x = 20
    line_h = 24
    # Semi-transparent background for readability
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 35), (min(w - 10, 10 + 580), 35 + 5 * line_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    status_text = "Recording" if is_recording else "Idle"
    cv2.putText(frame, f"Status: {status_text}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    y += line_h

    last_idx = max(0, next_index - 1)
    last_text = f"exp_{last_idx}.mp4" if last_idx > 0 else "-"
    cv2.putText(frame, f"Last: {last_text}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
    y += line_h

    cv2.putText(frame, f"Next: exp_{next_index}.mp4", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
    y += line_h

    cv2.putText(frame, f"Res: {color_resolution}  FPS: {int(fps)}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
    y += line_h

    # Show only tail of path if too long
    tail = output_dir if len(output_dir) < 50 else ("…" + output_dir[-49:])
    cv2.putText(frame, f"Dir: {tail}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)


def open_device(color_resolution: str):
    if PyK4A is None:
        print(
            "Failed to import pyk4a. Please install it: pip install pyk4a, and ensure Azure Kinect SDK is installed.",
            file=sys.stderr,
        )
        if _IMPORT_ERROR is not None:
            print(f"Import error: {_IMPORT_ERROR}", file=sys.stderr)
        sys.exit(1)

    resolution_map = {
        "720p": ColorResolution.RES_720P,
        "1080p": ColorResolution.RES_1080P,
        "1440p": ColorResolution.RES_1440P,
        "1536p": ColorResolution.RES_1536P,
        "2160p": ColorResolution.RES_2160P,
        "3072p": ColorResolution.RES_3072P,
    }

    if color_resolution not in resolution_map:
        print(
            f"Unknown color resolution '{color_resolution}'. Choose from: {', '.join(resolution_map.keys())}",
            file=sys.stderr,
        )
        sys.exit(2)

    # Choose a BGRA format compatible across pyk4a versions
    if ColorFormat is not None:
        color_format = getattr(ColorFormat, "COLOR_BGRA", None)
    else:
        color_format = None
    if color_format is None and ImageFormat is not None:
        color_format = getattr(ImageFormat, "COLOR_BGRA32", None)

    if color_format is None:
        print("Unable to select a BGRA color format for this pyk4a version.", file=sys.stderr)
        sys.exit(3)

    k4a = PyK4A(
        Config(
            color_resolution=resolution_map[color_resolution],
            color_format=color_format,
            synchronized_images_only=False,
        )
    )
    k4a.start()
    return k4a


def create_writer(path: str, width: int, height: int, fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    return writer


class WriterThread(threading.Thread):
    def __init__(self, fps: float, queue_size: int = 60):
        super().__init__(daemon=True)
        self.fps = fps
        self.queue: Queue = Queue(maxsize=queue_size)
        self._writer: Optional[cv2.VideoWriter] = None
        self._running = True

    def run(self) -> None:
        while self._running:
            try:
                item = self.queue.get(timeout=0.1)
            except Empty:
                continue

            itype = item[0]
            if itype == 'exit':
                break
            elif itype == 'start':
                _, path, width, height = item
                if self._writer is not None:
                    try:
                        self._writer.release()
                    except Exception:
                        pass
                    self._writer = None
                self._writer = create_writer(path, width, height, self.fps)
            elif itype == 'stop':
                if self._writer is not None:
                    try:
                        self._writer.release()
                    except Exception:
                        pass
                    self._writer = None
            elif itype == 'frame':
                _, frame = item
                if self._writer is not None:
                    try:
                        self._writer.write(frame)
                    except Exception:
                        # Drop frame on write error
                        pass

        # Cleanup
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
            self._writer = None

    def start_recording(self, path: str, width: int, height: int) -> None:
        self.queue.put(('start', path, width, height))

    def stop_recording(self) -> None:
        self.queue.put(('stop',))

    def enqueue_frame(self, frame) -> None:
        try:
            self.queue.put_nowait(('frame', frame))
        except Full:
            # Drop the frame to avoid blocking
            pass

    def shutdown(self) -> None:
        self._running = False
        try:
            self.queue.put_nowait(('exit',))
        except Full:
            # Best-effort
            pass


def main():
    parser = argparse.ArgumentParser(description="Azure Kinect viewer/recorder (press r to toggle record, q to quit)")
    parser.add_argument("--output-dir", default="/home/demo-panda/Videos/vc-umi", help="Base directory to save recordings")
    parser.add_argument("--task", default=None, help="Task name; creates and saves into a subdirectory of output-dir")
    parser.add_argument("--fps", type=float, default=30.0, help="Recording FPS (target)")
    parser.add_argument("--color-resolution", default="1440p", help="Color resolution: 720p, 1080p, 1440p, 1536p, 2160p, 3072p")
    parser.add_argument("--window", default="AzureKinect", help="OpenCV window name")
    args = parser.parse_args()

    effective_output_dir = args.output_dir
    if args.task:
        effective_output_dir = os.path.join(args.output_dir, args.task)
    ensure_output_dir(effective_output_dir)

    k4a = open_device(args.color_resolution)

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    # Get screen resolution and set window to a reasonable size
    screen_width = 1920  # Default fallback
    screen_height = 1080
    
    
    try:
        import tkinter as tk
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
    except:
        pass
    
    # Set window to about 80% of screen size
    window_width = int(screen_width * 0.8)
    window_height = int(screen_height * 0.8)
    cv2.resizeWindow(args.window, window_width, window_height)
    is_recording = False
    writer = None
    output_path = None
    record_start_time = None
    next_index = get_next_exp_index(effective_output_dir)

    # Background writer thread to avoid blocking the capture loop
    bg_writer = WriterThread(fps=args.fps, queue_size=90)
    bg_writer.start()

    try:
        last_time = time.time()
        while True:
            capture = k4a.get_capture()
            color = capture.color
            if color is None:
                continue

            # Convert to BGR for OpenCV if needed
            if color.ndim == 3 and color.shape[2] == 4:
                frame_bgr = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
            elif color.ndim == 3 and color.shape[2] == 3:
                frame_bgr = color
            else:
                # Unexpected format; try a safe conversion
                try:
                    frame_bgr = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
                except Exception:
                    frame_bgr = color

            if is_recording:
                # Enqueue a copy so on-screen overlays don't affect saved frames
                bg_writer.enqueue_frame(frame_bgr.copy())

            # Elapsed timer and episode name visualization when recording
            elapsed = None
            ep_name = None
            if is_recording and record_start_time is not None:
                elapsed = time.time() - record_start_time
                if output_path is not None:
                    ep_name = os.path.basename(output_path)
            draw_rec_indicator(frame_bgr, is_recording, elapsed, ep_name)
            draw_status_overlay(
                frame_bgr,
                is_recording,
                effective_output_dir,
                args.color_resolution,
                args.fps,
                next_index,
                os.path.basename(output_path) if output_path else None,
            )
            cv2.imshow(args.window, frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            # Detect window close (X button)
            window_visible = True
            try:
                window_visible = cv2.getWindowProperty(args.window, cv2.WND_PROP_VISIBLE) >= 1
            except Exception:
                window_visible = False

            want_quit = (key == ord('q'))
            want_soft_close = (key == ord('x')) or (not window_visible)
            if want_quit or want_soft_close:
                if is_recording:
                    # Ask user whether to save or discard current recording
                    prompt_frame = frame_bgr.copy()
                    overlay = prompt_frame.copy()
                    h, w = prompt_frame.shape[:2]
                    cv2.rectangle(overlay, (int(w*0.15), int(h*0.4)), (int(w*0.85), int(h*0.6)), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, prompt_frame, 0.4, 0, prompt_frame)
                    cv2.putText(prompt_frame, "Save current recording? (y/n)", (int(w*0.2), int(h*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
                    cv2.imshow(args.window, prompt_frame)

                    choice = None
                    while True:
                        k = cv2.waitKey(0) & 0xFF
                        if k in (ord('y'), ord('Y')):
                            choice = 'y'
                            break
                        if k in (ord('n'), ord('N')):
                            choice = 'n'
                            break
                        # If window was closed during prompt, treat as save to be safe
                        try:
                            if cv2.getWindowProperty(args.window, cv2.WND_PROP_VISIBLE) < 1:
                                choice = 'y'
                                break
                        except Exception:
                            choice = 'y'
                            break

                    if choice == 'y':
                        if want_quit:
                            # On explicit quit, stop and exit
                            bg_writer.stop_recording()
                            time.sleep(0.05)
                            print(f"Recording saved: {output_path}")
                            break
                        else:
                            # On window close/x, keep recording and do not exit
                            print("Continuing recording; window close canceled.")
                            # fall through to continue loop without stopping
                    else:
                        # Discard: stop and delete file; return to idle, preserve episode number for next start
                        bg_writer.stop_recording()
                        time.sleep(0.05)
                        try:
                            if output_path and os.path.isfile(output_path):
                                os.remove(output_path)
                                print(f"Recording discarded: {output_path}")
                        except Exception as e:
                            print(f"Failed to delete file: {output_path} ({e})")

                        try:
                            base = os.path.basename(output_path) if output_path else ''
                            idx = int(base.replace('exp_','').replace('.mp4','')) if base.startswith('exp_') else get_next_exp_index(effective_output_dir)
                            next_index = idx
                        except Exception:
                            next_index = get_next_exp_index(effective_output_dir)

                        is_recording = False
                        record_start_time = None
                        output_path = None
                        # Stay in app; user can press 'r' to start again
                else:
                    if want_quit:
                        break
            elif key == ord('r'):
                # Start or rollover to a new recording file
                if is_recording:
                    # Close current file and start a new incremented one
                    bg_writer.stop_recording()
                    print(f"Recording saved: {output_path}")
                    # increment next index before building next name
                    next_index = get_next_exp_index(effective_output_dir)
                    output_path = build_output_path(effective_output_dir)
                    height, width = frame_bgr.shape[:2]
                    bg_writer.start_recording(output_path, width, height)
                    record_start_time = time.time()
                    is_recording = True
                    print(f"Recording started: {output_path}")
                else:
                    next_index = get_next_exp_index(effective_output_dir)
                    output_path = build_output_path(effective_output_dir)
                    height, width = frame_bgr.shape[:2]
                    bg_writer.start_recording(output_path, width, height)
                    record_start_time = time.time()
                    is_recording = True
                    print(f"Recording started: {output_path}")
            elif key == ord('s'):
                # Stop and save
                if is_recording:
                    is_recording = False
                    record_start_time = None
                    bg_writer.stop_recording()
                    print(f"Recording saved: {output_path}")
                    output_path = None
                    next_index = get_next_exp_index(effective_output_dir)

            # Simple sleep to avoid maxing CPU if needed
            now = time.time()
            if now - last_time < 0.001:
                time.sleep(0.001)
            last_time = now

    except KeyboardInterrupt:
        pass
    finally:
        # Ensure writer thread stops and releases resources
        try:
            bg_writer.stop_recording()
        except Exception:
            pass
        try:
            bg_writer.shutdown()
            bg_writer.join(timeout=2.0)
        except Exception:
            pass
        try:
            k4a.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
