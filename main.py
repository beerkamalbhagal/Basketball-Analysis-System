from utils.video_utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer, ball_tracks_drawer

def main():
    # Read video   
    video_frames = read_video("input_videos/video_1.mp4")

    # Initialize trackers
    player_tracker = PlayerTracker("models/player_detector.pt")
    ball_tracker = BallTracker("models/ball_detector.pt")

    # Run trackers
    player_tracks = player_tracker.get_object_tracks(video_frames,
                                                     read_from_stub=True,
                                                    stub_path="stubs/player_tracks_stubs.pkl")
    
    ball_tracks = ball_tracker.get_object_tracks(video_frames,
                                                  read_from_stub=True,
                                                  stub_path="stubs/ball_tracks_stubs.pkl")
    
    # Remove wrong detections
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    # Interpolate ball positions
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    # Draw output
    # Initialize drawers
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()

    # Draw object tracks
    output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks)
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)
    # Save video   
    save_video(output_video_frames, "output_videos/output_video.avi") 

if __name__ == "__main__":
    main()