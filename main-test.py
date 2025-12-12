import cv2
import json
import os
from pathlib import Path
from ultralytics import solutions

class InteractiveObjectCounter:
    """Interactive Object Counter with mouse-based line drawing and coordinate persistence."""
    
    def __init__(self, video_path, model_path, classes_to_count, line_config_file="line_config.json"):
        """
        Initialize the Interactive Object Counter.
        
        Args:
            video_path (str): Path to input video file
            model_path (str): Path to YOLO model
            classes_to_count (list): List of class indices to count
            line_config_file (str): JSON file to save/load line coordinates
        """
        self.video_path = video_path
        self.model_path = model_path
        self.classes_to_count = classes_to_count
        self.line_config_file = line_config_file
        
        self.line_points = []
        self.drawing = False
        self.first_frame = None
        self.temp_frame = None
        
        # Load existing line coordinates if available
        self.load_line_config()
    
    def load_line_config(self):
        """Load line coordinates from JSON file if it exists."""
        if os.path.exists(self.line_config_file):
            try:
                with open(self.line_config_file, 'r') as f:
                    config = json.load(f)
                    video_name = Path(self.video_path).name
                    if video_name in config:
                        self.line_points = config[video_name]
                        print(f"✓ Loaded existing line coordinates: {self.line_points}")
                    else:
                        print(f"ℹ No saved line for this video: {video_name}")
            except Exception as e:
                print(f"⚠ Error loading line config: {e}")
        else:
            print(f"ℹ No line config file found. Will create new one.")
    
    def save_line_config(self):
        """Save line coordinates to JSON file."""
        try:
            # Load existing config or create new
            config = {}
            if os.path.exists(self.line_config_file):
                with open(self.line_config_file, 'r') as f:
                    config = json.load(f)
            
            # Update with current video's line
            video_name = Path(self.video_path).name
            config[video_name] = self.line_points
            
            # Save to file
            with open(self.line_config_file, 'w') as f:
                json.dump(config, f, indent=4)
            
            print(f"✓ Line coordinates saved to {self.line_config_file}")
        except Exception as e:
            print(f"⚠ Error saving line config: {e}")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing the counting line."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.line_points) < 2:
                self.line_points.append((x, y))
                print(f"Point {len(self.line_points)} selected: ({x}, {y})")
                
                # Draw point on frame
                cv2.circle(self.temp_frame, (x, y), 5, (0, 255, 0), -1)
                
                # If we have 2 points, draw the line
                if len(self.line_points) == 2:
                    cv2.line(self.temp_frame, self.line_points[0], self.line_points[1], (0, 255, 0), 2)
                    cv2.putText(self.temp_frame, "Press 'S' to Save or 'R' to Reset", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Draw Counting Line", self.temp_frame)
    
    def draw_line_interface(self, frame):
        """Interactive interface for drawing the counting line."""
        self.first_frame = frame.copy()
        self.temp_frame = frame.copy()
        
        cv2.namedWindow("Draw Counting Line")
        cv2.setMouseCallback("Draw Counting Line", self.mouse_callback)
        
        # Add instructions
        instructions = [
            "Instructions:",
            "1. Click TWO points to draw counting line",
            "2. Press 'S' to SAVE and continue",
            "3. Press 'R' to RESET points",
            "4. Press 'Q' to QUIT"
        ]
        
        y_offset = 50
        for i, instruction in enumerate(instructions):
            cv2.putText(self.temp_frame, instruction, (10, y_offset + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("Draw Counting Line", self.temp_frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') or key == ord('S'):
                if len(self.line_points) == 2:
                    print("✓ Line saved!")
                    self.save_line_config()
                    cv2.destroyWindow("Draw Counting Line")
                    return True
                else:
                    print("⚠ Please select 2 points before saving!")
            
            elif key == ord('r') or key == ord('R'):
                print("↻ Resetting points...")
                self.line_points = []
                self.temp_frame = self.first_frame.copy()
                
                # Redraw instructions
                y_offset = 50
                for i, instruction in enumerate(instructions):
                    cv2.putText(self.temp_frame, instruction, (10, y_offset + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                cv2.imshow("Draw Counting Line", self.temp_frame)
            
            elif key == ord('q') or key == ord('Q'):
                print("✗ Exiting without saving")
                cv2.destroyWindow("Draw Counting Line")
                return False
    
    def count_objects(self):
        """Count specific classes of objects in the video."""
        cap = cv2.VideoCapture(self.video_path)
        assert cap.isOpened(), "Error reading video file"
        
        # Get first frame for line drawing if needed
        ret, first_frame = cap.read()
        if not ret:
            print("Error reading first frame")
            return
        
        # If no line exists, draw one
        if len(self.line_points) != 2:
            print("\n📏 No counting line found. Please draw one...")
            if not self.draw_line_interface(first_frame):
                print("Line drawing cancelled. Exiting...")
                cap.release()
                return
        else:
            print(f"✓ Using existing line: {self.line_points}")
        
        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Initialize counter with the enhanced ObjectCounter class
        counter = solutions.ObjectCounter(
            show=True,
            region=self.line_points,
            model=self.model_path,
            classes=self.classes_to_count
        )
        
        print("\n▶ Processing video... Press 'Q' to quit")
        frame_count = 0
        
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("\n✓ Video processing complete!")
                break
            
            # Process frame and get results
            results = counter(im0)
            
            # Display frame count and counts on the frame
            frame_count += 1
            info_text = f"Frame: {frame_count} | IN: {counter.in_count} | OUT: {counter.out_count}"
            cv2.putText(results.plot_im, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show the frame
            cv2.imshow("Object Counting", results.plot_im)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⏹ Processing stopped by user")
                break
        
        # Print final statistics
        print("\n" + "="*50)
        print("FINAL STATISTICS")
        print("="*50)
        print(f"Total Frames Processed: {frame_count}")
        print(f"Objects IN: {counter.in_count}")
        print(f"Objects OUT: {counter.out_count}")
        print(f"Net Count: {counter.in_count - counter.out_count}")
        
        if hasattr(counter, 'classwise_count'):
            print("\nClass-wise Counts:")
            for class_name, counts in counter.classwise_count.items():
                print(f"  {class_name}: IN={counts['IN']}, OUT={counts['OUT']}")
        print("="*50)
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Main function to run the interactive object counter."""
    
    # Configuration
    video_path = "path/to/video.mp4"
    model_path = "yolo11n.pt"
    classes_to_count = [0, 2]  # 0=person, 2=car in COCO dataset
    
    # Create counter instance
    counter = InteractiveObjectCounter(
        video_path=video_path,
        model_path=model_path,
        classes_to_count=classes_to_count,
        line_config_file="line_config.json"
    )
    
    # Run counting
    counter.count_objects()


if __name__ == "__main__":
    main()
