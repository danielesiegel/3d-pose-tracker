#!/usr/bin/env python3
"""
3D Pose Viewer for replaying saved Parquet data.
Loads and displays captured pose data in an interactive 3D viewer.
"""

import click
import pandas as pd
import numpy as np
import open3d as o3d
import time
from pathlib import Path


# Joint colors (RGB)
COLORS = {
    'shoulder': [1.0, 0.0, 0.0],  # Red
    'elbow': [0.0, 1.0, 0.0],     # Green
    'wrist': [0.0, 0.0, 1.0]      # Blue
}


class PoseViewer:
    """Interactive 3D viewer for pose data."""

    def __init__(self, data_file, playback_speed=1.0, loop=False):
        self.data_file = Path(data_file)
        self.playback_speed = playback_speed
        self.loop = loop

        # Load data
        self.df = None
        self.load_data()

    def load_data(self):
        """Load pose data from Parquet file."""
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

        click.echo(f"Loading data from {self.data_file}...")
        self.df = pd.read_parquet(self.data_file, engine='pyarrow')

        click.echo(f"Loaded {len(self.df)} frames")
        click.echo(f"Columns: {list(self.df.columns)}")

        # Validate required columns
        required_cols = [
            'shoulder_x', 'shoulder_y', 'shoulder_z',
            'elbow_x', 'elbow_y', 'elbow_z',
            'wrist_x', 'wrist_y', 'wrist_z'
        ]

        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def get_frame_data(self, frame_idx):
        """Extract joint positions for a specific frame."""
        row = self.df.iloc[frame_idx]

        points = np.array([
            [row['shoulder_x'], row['shoulder_y'], row['shoulder_z']],
            [row['elbow_x'], row['elbow_y'], row['elbow_z']],
            [row['wrist_x'], row['wrist_y'], row['wrist_z']]
        ])

        colors = np.array([
            COLORS['shoulder'],
            COLORS['elbow'],
            COLORS['wrist']
        ])

        return points, colors

    def create_trajectory(self):
        """Create trajectory lines showing motion path for each joint."""
        # Extract all positions
        shoulder_traj = self.df[['shoulder_x', 'shoulder_y', 'shoulder_z']].values
        elbow_traj = self.df[['elbow_x', 'elbow_y', 'elbow_z']].values
        wrist_traj = self.df[['wrist_x', 'wrist_y', 'wrist_z']].values

        trajectories = []

        # Create line set for each joint trajectory
        for traj, color in [
            (shoulder_traj, COLORS['shoulder']),
            (elbow_traj, COLORS['elbow']),
            (wrist_traj, COLORS['wrist'])
        ]:
            if len(traj) > 1:
                lines = [[i, i + 1] for i in range(len(traj) - 1)]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(traj)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])
                trajectories.append(line_set)

        return trajectories

    def run_interactive(self):
        """Run interactive viewer with playback controls."""
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Pose Viewer - Press SPACE to play/pause, R to reset",
                         width=1024, height=768)

        # Create geometries
        pcd = o3d.geometry.PointCloud()
        bones = o3d.geometry.LineSet()

        # Add geometries
        vis.add_geometry(pcd)
        vis.add_geometry(bones)

        # Add trajectory
        trajectories = self.create_trajectory()
        for traj in trajectories:
            vis.add_geometry(traj)

        # Set up rendering options
        opt = vis.get_render_option()
        opt.point_size = 20.0
        opt.line_width = 5.0
        opt.background_color = np.array([0.05, 0.05, 0.05])

        # Playback state
        state = {
            'frame': 0,
            'playing': True,
            'last_update': time.time()
        }

        def toggle_play(vis):
            state['playing'] = not state['playing']
            status = "Playing" if state['playing'] else "Paused"
            click.echo(f"\n{status} at frame {state['frame']}/{len(self.df)}")

        def reset_view(vis):
            state['frame'] = 0
            click.echo(f"\nReset to frame 0")

        # Register key callbacks
        vis.register_key_callback(32, toggle_play)  # SPACE
        vis.register_key_callback(ord('R'), reset_view)

        # Initialize with first frame
        points, colors = self.get_frame_data(0)
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Create bones
        bone_lines = [[0, 1], [1, 2]]  # shoulder->elbow, elbow->wrist
        bones.points = o3d.utility.Vector3dVector(points)
        bones.lines = o3d.utility.Vector2iVector(bone_lines)
        bones.colors = o3d.utility.Vector3dVector([[1, 1, 1], [1, 1, 1]])

        vis.update_geometry(pcd)
        vis.update_geometry(bones)
        vis.reset_view_point(True)

        click.echo("\n=== Controls ===")
        click.echo("SPACE: Play/Pause")
        click.echo("R: Reset to beginning")
        click.echo("Mouse: Rotate view")
        click.echo("Close window or press ESC to exit")
        click.echo(f"\nPlayback speed: {self.playback_speed}x")
        click.echo(f"Total frames: {len(self.df)}")
        click.echo(f"Loop: {'Enabled' if self.loop else 'Disabled'}\n")

        # Main loop
        while vis.poll_events():
            current_time = time.time()

            if state['playing']:
                # Calculate expected frame interval
                fps = 30  # Assume original capture was 30fps
                frame_interval = (1.0 / fps) / self.playback_speed

                if current_time - state['last_update'] >= frame_interval:
                    # Update to next frame
                    state['frame'] = (state['frame'] + 1) % len(self.df)

                    # Stop if not looping
                    if state['frame'] == 0 and not self.loop:
                        state['playing'] = False
                        click.echo("\nPlayback finished")

                    # Update geometry
                    points, colors = self.get_frame_data(state['frame'])

                    pcd.points = o3d.utility.Vector3dVector(points)
                    pcd.colors = o3d.utility.Vector3dVector(colors)

                    bones.points = o3d.utility.Vector3dVector(points)

                    vis.update_geometry(pcd)
                    vis.update_geometry(bones)

                    state['last_update'] = current_time

            vis.update_renderer()

        vis.destroy_window()

    def run_static(self):
        """Display static view with full trajectory."""
        click.echo("Displaying static trajectory view...")

        geometries = []

        # Create trajectory
        trajectories = self.create_trajectory()
        geometries.extend(trajectories)

        # Add start and end point markers
        start_points, start_colors = self.get_frame_data(0)
        end_points, end_colors = self.get_frame_data(len(self.df) - 1)

        # Start marker (larger, semi-transparent)
        start_pcd = o3d.geometry.PointCloud()
        start_pcd.points = o3d.utility.Vector3dVector(start_points)
        start_pcd.colors = o3d.utility.Vector3dVector(start_colors * 0.5)
        geometries.append(start_pcd)

        # End marker
        end_pcd = o3d.geometry.PointCloud()
        end_pcd.points = o3d.utility.Vector3dVector(end_points)
        end_pcd.colors = o3d.utility.Vector3dVector(end_colors)
        geometries.append(end_pcd)

        # Create coordinate frame at origin
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geometries.append(coord_frame)

        # Display
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Pose Trajectory (Static View)",
            width=1024,
            height=768,
            point_show_normal=False
        )


@click.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--speed', '-s', default=1.0, type=float,
              help='Playback speed multiplier (default: 1.0)')
@click.option('--loop', '-l', is_flag=True,
              help='Loop playback continuously')
@click.option('--static', is_flag=True,
              help='Show static trajectory view instead of animated playback')
def main(data_file, speed, loop, static):
    """
    View saved pose data from Parquet file.

    DATA_FILE: Path to the Parquet file containing pose data.

    Interactive mode (default):
    - SPACE: Play/Pause
    - R: Reset to beginning
    - Mouse: Rotate view

    Static mode (--static flag):
    - Shows full trajectory path
    - Displays start and end positions
    """
    try:
        viewer = PoseViewer(data_file, playback_speed=speed, loop=loop)

        if static:
            viewer.run_static()
        else:
            viewer.run_interactive()

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise


if __name__ == "__main__":
    main()
