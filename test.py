def check_lost_ids(self):
    """
    Check for objects that disappeared without being counted.
    
    FIXED LOGIC:
    - If crossed line & came from IN side (s2 should end > 0) → Missed IN
    - If crossed line & came from OUT side (s2 should end < 0) → Missed OUT  
    - If never crossed → Missed Cross
    """
    current = self.frame_count
    lost = []

    for tid, last in self.last_seen.items():
        if current - last > self.max_missing_frames:
            lost.append(tid)

    for tid in lost:
        # Case 1: Object crossed the line but wasn't counted
        if tid in self.crossed_ids and tid not in self.counted:
            # Get last known position to determine expected direction
            if tid in self.hist:
                last_cx, last_cy = self.hist[tid]
                last_side = self.side(last_cx, last_cy, *self.line_p1, *self.line_p2)
                
                # IN logic: Should have ended on positive side (s2 > 0)
                if last_side > 0:
                    self.missed_in.add(tid)
                    print(f"⚠️ MISSED IN - ID:{tid} (crossed line, ended on IN side, not counted)")
                
                # OUT logic: Should have ended on negative side (s2 < 0)  
                elif last_side < 0:
                    self.missed_out.add(tid)
                    print(f"⚠️ MISSED OUT - ID:{tid} (crossed line, ended on OUT side, not counted)")
            else:
                # Fallback to origin if no position data
                origin = self.origin_side.get(tid, "UNKNOWN")
                if origin == "IN":
                    self.missed_in.add(tid)
                elif origin == "OUT":
                    self.missed_out.add(tid)
        
        # Case 2: Object never crossed the line
        elif tid not in self.counted and tid not in self.crossed_ids:
            self.missed_cross.add(tid)
            print(f"⚠️ MISSED CROSS - ID:{tid} (disappeared without crossing)")

        # Cleanup
        self.hist.pop(tid, None)
        self.last_seen.pop(tid, None)
        self.color_at_crossing.pop(tid, None)
        self.origin_side.pop(tid, None)
