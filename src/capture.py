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
import open3d as o3d
from datetime import datetime
from pathlib import Path
import threading
from queue import Queue


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

        # 3D visualization setup
        self.vis_queue = Queue(maxsize=2) if enable_3d else None
        self.running = True
        self.vis_thread = None

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

    def visualize_3d(self):
        """Run 3D visualization in separate thread."""
        try:
            vis = o3d.visualization.Visualizer()

            # Try to create window with error handling
            try:
                vis.create_window(window_name="3D Pose View", width=640, height=480)
            except Exception as e:
                click.echo(f"Warning: Could not create 3D window: {e}")
                click.echo("Continuing without 3D visualization...")
                return

            # Create point cloud for joints
            pcd = o3d.geometry.PointCloud()

            # Create line set for bones
            lines = o3d.geometry.LineSet()

            # Initialize geometry
            vis.add_geometry(pcd)
            vis.add_geometry(lines)

            # Set up view
            opt = vis.get_render_option()
            opt.point_size = 15.0
            opt.background_color = np.array([0.1, 0.1, 0.1])

            # Set up camera for better initial view
            ctr = vis.get_view_control()

            first_update = True

            while self.running:
                if not self.vis_queue.empty():
                    points_data = self.vis_queue.get()

                    if points_data is not None:
                        points = points_data['points']
                        colors = points_data['colors']

                        # Update point cloud
                        pcd.points = o3d.utility.Vector3dVector(points)
                        pcd.colors = o3d.utility.Vector3dVector(colors)

                        # Update lines (bones)
                        line_points = points
                        line_indices = [[0, 1], [1, 2]]  # shoulder->elbow, elbow->wrist
                        lines.points = o3d.utility.Vector3dVector(line_points)
                        lines.lines = o3d.utility.Vector2iVector(line_indices)
                        lines.colors = o3d.utility.Vector3dVector([[1, 1, 1], [1, 1, 1]])

                        vis.update_geometry(pcd)
                        vis.update_geometry(lines)

                        if first_update:
                            vis.reset_view_point(True)
                            first_update = False

                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.01)

            vis.destroy_window()
        except Exception as e:
            click.echo(f"Warning: 3D visualization error: {e}")
            click.echo("Continuing with 2D capture only...")

    def run(self):
        """Main capture loop."""
        # Start 3D visualization thread if enabled
        if self.enable_3d:
            self.vis_thread = threading.Thread(target=self.visualize_3d, daemon=True)
            self.vis_thread.start()
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
                        if self.enable_3d and self.vis_queue is not None:
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

                            # Send to visualization queue (non-blocking)
                            if self.vis_queue.full():
                                try:
                                    self.vis_queue.get_nowait()
                                except:
                                    pass

                            self.vis_queue.put({'points': points, 'colors': colors})

                # Draw 2D overlay
                frame = self.draw_2d_overlay(frame, results)

                # Add FPS and frame count
                cv2.putText(frame, f"FPS: {self.fps} | Frames: {self.frame_count}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                self.frame_count += 1
                last_process_time = current_time

            # Display frame
            cv2.imshow('Pose Capture', frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        self.running = False
        cap.release()
        cv2.destroyAllWindows()
        self.pose.close()

        # Wait for visualization thread if it was started
        if self.vis_thread is not None:
            self.vis_thread.join(timeout=2.0)

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
