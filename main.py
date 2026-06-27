from utils.video_utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer, TeamBallControlDrawer, PassInterceptionDrawer
from team_assigner import TeamAssigner
from ball_acquisition import BallAcquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector

def main():
    # Read video   
    video_frames = read_video("input_videos/video_2.mp4")

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

    # Assign player to teams
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_team_across_frames(video_frames, player_tracks, read_from_stub = True, stub_path="stubs/player_assignment_stub.pkl")

    # Ball Possession
    ball_acquisition_detector = BallAcquisitionDetector()
    ball_acquisition = ball_acquisition_detector.detect_ball_possession(player_tracks, ball_tracks)

    # Passes and Interceptions
    pass_and_interception_detector = PassAndInterceptionDetector()
    passes = pass_and_interception_detector.detect_passes(ball_acquisition, player_assignment)
    interceptions = pass_and_interception_detector.detect_interceptions(ball_acquisition, player_assignment)

      # Draw output
    # Initialize drawers
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    team_ball_control_drawer = TeamBallControlDrawer()
    pass_interception_drawer = PassInterceptionDrawer()

    # Draw object tracks
    output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks, player_assignment, ball_acquisition)
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)
    output_video_frames = team_ball_control_drawer.draw(output_video_frames, player_assignment, ball_acquisition)
    output_video_frames = pass_interception_drawer.draw(output_video_frames, passes, interceptions)
    output_video_frames = pass_interception_drawer.draw(output_video_frames, passes, interceptions)

    # Save video   
    save_video(output_video_frames, "output_videos/output_video.avi") 

if __name__ == "__main__":
    main()