#!/usr/bin/env python3
"""
Real-time 3D Pose Capture using MediaPipe BlazePose.
Captures left arm joints (shoulder, elbow, wrist) and saves to Parquet.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import click
import time
from datetime import datetime
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better macOS compatibility


# MediaPipe BlazePose landmark indices for left arm
LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15

# Colors for visualization (BGR for OpenCV, RGB for Open3D)
COLORS = {
    LEFT_SHOULDER: {'name': 'shoulder', 'bgr': (0, 0, 255), 'rgb': [1.0, 0.0, 0.0]},  # Red
    LEFT_ELBOW: {'name': 'elbow', 'bgr': (0, 255, 0), 'rgb': [0.0, 1.0, 0.0]},        # Green
    LEFT_WRIST: {'name': 'wrist', 'bgr': (255, 0, 0), 'rgb': [0.0, 0.0, 1.0]}         # Blue
}


class PoseCapture:
    """Handles pose capture and visualization."""

    def __init__(self, fps=30, output_dir="output", enable_3d=True):
        self.fps = fps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.enable_3d = enable_3d

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Use heavy model for better 3D accuracy
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Data storage
        self.pose_data = []
        self.frame_count = 0

        # 3D visualization setup (will be initialized on main thread)
        self.fig = None
        self.ax = None
        self.points_plot = None
        self.lines_plot = None

        # Trail effect: store last 3 seconds of frames
        self.trail_duration = 3.0  # seconds
        self.trail_buffer = deque(maxlen=int(fps * self.trail_duration))

        # Axis bounds tracking (only expand, never shrink)
        self.axis_bounds = {
            'x_min': -0.5, 'x_max': 0.5,
            'y_min': -0.5, 'y_max': 0.5,
            'z_min': -0.5, 'z_max': 0.5
        }

        # Quit flag for keyboard handling
        self.should_quit = False

    def process_frame(self, frame):
        """Process a single frame with MediaPipe."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Process with MediaPipe
        results = self.pose.process(rgb_frame)

        rgb_frame.flags.writeable = True

        return results

    def extract_left_arm_data(self, results, timestamp):
        """Extract left arm joint positions from MediaPipe results."""
        if not results.pose_world_landmarks:
            return None

        landmarks = results.pose_world_landmarks.landmark

        # Extract 3D positions for left arm joints
        data = {
            'timestamp': timestamp,
            'frame': self.frame_count,
            'shoulder_x': landmarks[LEFT_SHOULDER].x,
            'shoulder_y': landmarks[LEFT_SHOULDER].y,
            'shoulder_z': landmarks[LEFT_SHOULDER].z,
            'elbow_x': landmarks[LEFT_ELBOW].x,
            'elbow_y': landmarks[LEFT_ELBOW].y,
            'elbow_z': landmarks[LEFT_ELBOW].z,
            'wrist_x': landmarks[LEFT_WRIST].x,
            'wrist_y': landmarks[LEFT_WRIST].y,
            'wrist_z': landmarks[LEFT_WRIST].z,
        }

        return data

    def draw_2d_overlay(self, frame, results):
        """Draw 2D pose overlay on the frame."""
        if not results.pose_landmarks:
            return frame

        landmarks = results.pose_landmarks.landmark
        h, w = frame.shape[:2]

        # Draw only left arm joints
        for idx in [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST]:
            lm = landmarks[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            color = COLORS[idx]['bgr']
            cv2.circle(frame, (cx, cy), 8, color, -1)
            cv2.putText(frame, COLORS[idx]['name'], (cx + 10, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw connections
        def draw_line(idx1, idx2):
            lm1, lm2 = landmarks[idx1], landmarks[idx2]
            x1, y1 = int(lm1.x * w), int(lm1.y * h)
            x2, y2 = int(lm2.x * w), int(lm2.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        draw_line(LEFT_SHOULDER, LEFT_ELBOW)
        draw_line(LEFT_ELBOW, LEFT_WRIST)

        return frame

    def init_3d_plot(self):
        """Initialize 3D matplotlib plot (must be called on main thread)."""
        try:
            plt.ion()  # Interactive mode
            self.fig = plt.figure(figsize=(8, 6))
            self.ax = self.fig.add_subplot(111, projection='3d')

            # Set labels and title
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title('3D Pose View - Left Arm (Press Q to quit)')

            # Set background color
            self.ax.set_facecolor('#1a1a1a')
            self.fig.patch.set_facecolor('#2a2a2a')

            # Set initial view limits
            self.ax.set_xlim([self.axis_bounds['x_min'], self.axis_bounds['x_max']])
            self.ax.set_ylim([self.axis_bounds['y_min'], self.axis_bounds['y_max']])
            self.ax.set_zlim([self.axis_bounds['z_min'], self.axis_bounds['z_max']])

            # Add keyboard event handler
            def on_key(event):
                if event.key == 'q':
                    self.should_quit = True

            self.fig.canvas.mpl_connect('key_press_event', on_key)

            click.echo("3D visualization window opened (matplotlib)")
            click.echo("Press 'q' in either window to quit and save")
            return True
        except Exception as e:
            click.echo(f"Warning: Could not create 3D window: {e}")
            click.echo("Continuing without 3D visualization...")
            return False

    def update_3d_plot(self, points, colors):
        """Update 3D plot with new joint positions and trail effect."""
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            return False

        try:
            # Add current frame to trail buffer
            self.trail_buffer.append({'points': points.copy(), 'colors': colors.copy()})

            # Clear the entire axis
            self.ax.cla()

            # Reset labels and colors after clearing
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title('3D Pose View - Left Arm (Press Q to quit)')
            self.ax.set_facecolor('#1a1a1a')

            # Update axis bounds (only expand, never shrink)
            margin = 0.2
            xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]

            # Check if we need to expand bounds
            if xs.min() - margin < self.axis_bounds['x_min']:
                self.axis_bounds['x_min'] = xs.min() - margin
            if xs.max() + margin > self.axis_bounds['x_max']:
                self.axis_bounds['x_max'] = xs.max() + margin

            if ys.min() - margin < self.axis_bounds['y_min']:
                self.axis_bounds['y_min'] = ys.min() - margin
            if ys.max() + margin > self.axis_bounds['y_max']:
                self.axis_bounds['y_max'] = ys.max() + margin

            if zs.min() - margin < self.axis_bounds['z_min']:
                self.axis_bounds['z_min'] = zs.min() - margin
            if zs.max() + margin > self.axis_bounds['z_max']:
                self.axis_bounds['z_max'] = zs.max() + margin

            # Set axis limits (stable, only expanding)
            self.ax.set_xlim([self.axis_bounds['x_min'], self.axis_bounds['x_max']])
            self.ax.set_ylim([self.axis_bounds['y_min'], self.axis_bounds['y_max']])
            self.ax.set_zlim([self.axis_bounds['z_min'], self.axis_bounds['z_max']])

            # Draw trail frames (oldest to newest, so newest is on top)
            num_frames = len(self.trail_buffer)
            for i, frame_data in enumerate(self.trail_buffer):
                frame_points = frame_data['points']
                frame_colors = frame_data['colors']

                # Calculate alpha based on position in buffer (older = more transparent)
                # Alpha goes from ~0.1 (oldest) to 1.0 (newest)
                alpha = (i + 1) / num_frames
                alpha = 0.1 + (alpha * 0.9)  # Scale to 0.1-1.0 range

                # Extract coordinates
                frame_xs = frame_points[:, 0]
                frame_ys = frame_points[:, 1]
                frame_zs = frame_points[:, 2]

                # For current frame (newest), use full opacity and larger size
                if i == num_frames - 1:
                    # Current frame - full opacity, larger markers
                    self.ax.scatter(frame_xs, frame_ys, frame_zs, c=frame_colors,
                                   s=200, marker='o', edgecolors='white',
                                   linewidths=2, alpha=1.0, depthshade=False)

                    # Draw bones for current frame with full opacity
                    self.ax.plot([frame_points[0, 0], frame_points[1, 0]],
                                [frame_points[0, 1], frame_points[1, 1]],
                                [frame_points[0, 2], frame_points[1, 2]],
                                'w-', linewidth=3, alpha=1.0)
                    self.ax.plot([frame_points[1, 0], frame_points[2, 0]],
                                [frame_points[1, 1], frame_points[2, 1]],
                                [frame_points[1, 2], frame_points[2, 2]],
                                'w-', linewidth=3, alpha=1.0)
                else:
                    # Trail frames - fading opacity, smaller markers
                    size = 50 + (alpha * 150)  # Size from 50 to 200
                    self.ax.scatter(frame_xs, frame_ys, frame_zs, c=frame_colors,
                                   s=size, marker='o', alpha=alpha, depthshade=False)

                    # Draw bones for trail frames with fading opacity
                    line_width = 1 + (alpha * 2)  # Line width from 1 to 3
                    self.ax.plot([frame_points[0, 0], frame_points[1, 0]],
                                [frame_points[0, 1], frame_points[1, 1]],
                                [frame_points[0, 2], frame_points[1, 2]],
                                'w-', linewidth=line_width, alpha=alpha * 0.7)
                    self.ax.plot([frame_points[1, 0], frame_points[2, 0]],
                                [frame_points[1, 1], frame_points[2, 1]],
                                [frame_points[1, 2], frame_points[2, 2]],
                                'w-', linewidth=line_width, alpha=alpha * 0.7)

            # Update the figure
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

            return True
        except Exception as e:
            click.echo(f"Warning: Error updating 3D plot: {e}")
            return False

    def run(self):
        """Main capture loop (runs on main thread for macOS compatibility)."""
        # Initialize 3D visualization on main thread if enabled
        if self.enable_3d:
            if not self.init_3d_plot():
                self.enable_3d = False
        else:
            click.echo("3D visualization disabled")

        # Open webcam
        cap = cv2.VideoCapture(0)

        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not cap.isOpened():
            click.echo("Error: Could not open webcam", err=True)
            return

        # Calculate frame interval
        frame_interval = 1.0 / self.fps
        last_process_time = 0

        click.echo(f"Starting pose capture at {self.fps} FPS")
        click.echo("Press 'q' to quit and save data")

        try:
            while True:
                current_time = time.time()

                # Read frame
                ret, frame = cap.read()
                if not ret:
                    click.echo("Error: Failed to read frame", err=True)
                    break

                # Process at specified FPS
                if current_time - last_process_time >= frame_interval:
                    # Process pose
                    results = self.process_frame(frame)

                    # Extract and store data
                    if results.pose_world_landmarks:
                        timestamp = datetime.now().isoformat()
                        data = self.extract_left_arm_data(results, timestamp)

                        if data:
                            self.pose_data.append(data)

                            # Update 3D visualization if enabled
                            if self.enable_3d:
                                points = np.array([
                                    [data['shoulder_x'], data['shoulder_y'], data['shoulder_z']],
                                    [data['elbow_x'], data['elbow_y'], data['elbow_z']],
                                    [data['wrist_x'], data['wrist_y'], data['wrist_z']]
                                ])

                                colors = np.array([
                                    COLORS[LEFT_SHOULDER]['rgb'],
                                    COLORS[LEFT_ELBOW]['rgb'],
                                    COLORS[LEFT_WRIST]['rgb']
                                ])

                                # Update 3D plot on main thread
                                if not self.update_3d_plot(points, colors):
                                    self.enable_3d = False  # Disable if window was closed

                    # Draw 2D overlay
                    frame = self.draw_2d_overlay(frame, results)

                    # Add FPS and frame count
                    cv2.putText(frame, f"FPS: {self.fps} | Frames: {self.frame_count}",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    self.frame_count += 1
                    last_process_time = current_time

                # Display frame
                cv2.imshow('Pose Capture', frame)

                # Check for quit from either window
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or self.should_quit:
                    click.echo("\nQuitting and saving...")
                    break

                # Check if 3D window was closed
                if self.enable_3d and self.fig is not None:
                    if not plt.fignum_exists(self.fig.number):
                        click.echo("3D window closed, continuing with 2D only")
                        self.enable_3d = False

        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.pose.close()

            # Close matplotlib if open
            if self.fig is not None:
                plt.close(self.fig)

            # Save data
            self.save_data()

    def save_data(self):
        """Save captured data to Parquet file."""
        if not self.pose_data:
            click.echo("No data captured!")
            return

        df = pd.DataFrame(self.pose_data)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"pose_data_{timestamp}.parquet"

        # Save to Parquet
        df.to_parquet(filename, engine='pyarrow', compression='snappy', index=False)

        click.echo(f"\nSaved {len(df)} frames to {filename}")
        click.echo(f"Data shape: {df.shape}")
        click.echo(f"Duration: {len(df) / self.fps:.2f} seconds")


@click.command()
@click.option('--frames', '-f', default=30, type=int,
              help='Frames per second to capture (default: 30)')
@click.option('--output', '-o', default='output', type=str,
              help='Output directory for saved data (default: output)')
@click.option('--no-3d', is_flag=True,
              help='Disable 3D visualization (use if experiencing display issues)')
def main(frames, output, no_3d):
    """
    Capture 3D pose data from webcam using MediaPipe BlazePose.

    Tracks left arm joints (shoulder, elbow, wrist) and displays them
    in both 2D video feed and 3D spatial viewer. Press 'q' to quit and save.
    """
    if frames <= 0:
        click.echo("Error: --frames must be positive", err=True)
        return

    capture = PoseCapture(fps=frames, output_dir=output, enable_3d=not no_3d)

    try:
        capture.run()
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise


if __name__ == "__main__":
    main()
