import json
import os
import threading
import time
import queue
import cv2
try:
    from ultralytics import YOLO  # Pre-import to avoid thread lock
except ImportError:
    pass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import main as pipeline_main

# Define paths
ROOT_DIR = Path(__file__).resolve().parent
LOGS_DIR = ROOT_DIR / "logs"
DASHBOARD_DIR = ROOT_DIR / "dashboard"
INPUT_DIR = ROOT_DIR / "input"

# Global state for streaming and status
processing_status = "IDLE"
current_run_id = ""
frame_queue = queue.Queue(maxsize=10)

def pipeline_thread_target(filepath, source_name):
    global processing_status, current_run_id
    processing_status = "RUNNING"
    current_run_id = source_name
    
    def frame_callback(frame):
        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            # We don't want to block indefinitely if the stream isn't being consumed
            try:
                # empty queue to keep only latest frame to reduce latency
                while not frame_queue.empty():
                    frame_queue.get_nowait()
                frame_queue.put_nowait(buffer.tobytes())
            except queue.Full:
                pass
                
    try:
        pipeline_main.run_pipeline(source=str(filepath), config_path=None, frame_callback=frame_callback)
    except Exception as e:
        print(f"Pipeline error: {e}")
    finally:
        processing_status = "DONE"
        # Push a None or dummy to unblock stream
        try:
            while not frame_queue.empty():
                frame_queue.get_nowait()
            frame_queue.put_nowait(None)
        except queue.Full:
            pass

class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def do_POST(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path == "/api/upload":
            qs = parse_qs(parsed_path.query)
            original_filename = qs.get("filename", ["uploaded_video.mp4"])[0]
            INPUT_DIR.mkdir(exist_ok=True)
            
            # Save to a unique filename to prevent overwriting existing files
            timestamp = int(time.time())
            filename = f"upload_{timestamp}_{original_filename}"
            filepath = INPUT_DIR / filename
            
            length = int(self.headers.get("Content-Length", 0))
            if length > 0:
                with open(filepath, "wb") as f:
                    bytes_read = 0
                    while bytes_read < length:
                        chunk = self.rfile.read(min(8192*4, length - bytes_read))
                        if not chunk: break
                        f.write(chunk)
                        bytes_read += len(chunk)
            
            # Start pipeline thread
            source_name = filename.rsplit('.', 1)[0]
            processing_status = "RUNNING"
            current_run_id = source_name
            threading.Thread(target=pipeline_thread_target, args=(filepath, source_name), daemon=True).start()
            
            response_body = json.dumps({"status": "started"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == "/api/runs":
            self.serve_api_runs()
        elif parsed_path.path == "/api/logs":
            qs = parse_qs(parsed_path.query)
            run_id = qs.get("run", [None])[0]
            self.serve_api_logs(run_id)
        elif parsed_path.path == "/api/status":
            response_data = {"status": processing_status, "run_id": current_run_id}
            response_body = json.dumps(response_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        elif parsed_path.path == "/api/stream":
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            
            while processing_status == "RUNNING":
                try:
                    frame = frame_queue.get(timeout=1.0)
                    if frame is None:
                        break
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
                except queue.Empty:
                    pass
                except Exception as e:
                    print(f"Stream interrupted: {e}")
                    break
        else:
            super().do_GET()

    def serve_api_runs(self):
        runs = set()
        if LOGS_DIR.exists():
            for log_file in LOGS_DIR.glob("*.json"):
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if data and len(data) > 0 and "source" in data[0]:
                            runs.add(data[0]["source"])
                        else:
                            runs.add(log_file.stem.split("_")[0])
                except Exception:
                    pass
        
        response_data = {"runs": list(runs)}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response_data).encode("utf-8"))

    def serve_api_logs(self, filter_run_id=None):
        total_scans = 0
        total_violations = 0
        rule_breakdown = {
            "glove": {"status": "PASS", "detail": "Meets standard", "failed_count": 0},
            "hairnet": {"status": "PASS", "detail": "Meets standard", "failed_count": 0},
            "pest": {"status": "PASS", "detail": "Meets standard", "failed_count": 0}
        }
        
        chef_stats = {}
        pest_events = []

        if LOGS_DIR.exists():
            for log_file in LOGS_DIR.glob("*.json"):
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        
                        if not data:
                            continue
                            
                        source = data[0].get("source", "")
                        if filter_run_id and source != filter_run_id:
                            continue
                            
                        total_scans += 1
                        
                        for event in data:
                            if "violation_type" in event:
                                v_type = event["violation_type"]
                                total_violations += 1
                                
                                if "glove" in v_type:
                                    rule_breakdown["glove"]["status"] = "FAIL"
                                    rule_breakdown["glove"]["failed_count"] += 1
                                elif "hairnet" in v_type:
                                    rule_breakdown["hairnet"]["status"] = "FAIL"
                                    rule_breakdown["hairnet"]["failed_count"] += 1
                                elif "pest" in v_type:
                                    rule_breakdown["pest"]["status"] = "FAIL"
                                    rule_breakdown["pest"]["failed_count"] += 1
                                    
                                    pest_events.append({
                                        "timestamp": event.get("timestamp"),
                                        "confidence": event.get("confidence", 0.0),
                                        "duration": event.get("duration_seconds", 0.0)
                                    })

                                track_id = event.get("track_id")
                                if track_id is not None:
                                    tid = str(track_id)
                                    if tid not in chef_stats:
                                        chef_stats[tid] = {
                                            "track_id": track_id,
                                            "violations": 0,
                                            "duration_sum": 0.0,
                                            "confidences": [],
                                            "labels": []
                                        }
                                    
                                    chef_stats[tid]["violations"] += 1
                                    chef_stats[tid]["duration_sum"] += event.get("duration_seconds", 0.0)
                                    chef_stats[tid]["labels"].append(v_type)
                                    
                                    if "confidence" in event:
                                        chef_stats[tid]["confidences"].append(event["confidence"])

                except Exception as e:
                    print(f"Error reading {log_file}: {e}")

        for rule, data in rule_breakdown.items():
            if data["status"] == "FAIL":
                data["detail"] = f"Non-conformance detected ({data['failed_count']} times) — remediation required."

        for tid, stats in chef_stats.items():
            confs = stats["confidences"]
            stats["avg_confidence"] = round(sum(confs) / len(confs), 4) if confs else 0.0
            stats["labels"] = list(set(stats["labels"]))
            stats["duration_sum"] = round(stats["duration_sum"], 2)
            del stats["confidences"]

        rules_passed = sum(1 for v in rule_breakdown.values() if v["status"] == "PASS")
        total_rules = len(rule_breakdown)
        compliance_score = int((rules_passed / total_rules) * 100) if total_rules > 0 else 100
        
        response_data = {
            "total_scans": total_scans,
            "compliance_score": compliance_score,
            "total_violations": total_violations,
            "rules_passed": rules_passed,
            "rules_failed": total_rules - rules_passed,
            "rule_breakdown": rule_breakdown,
            "chef_analytics": list(chef_stats.values()),
            "pest_analytics": pest_events
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response_data).encode("utf-8"))

if __name__ == "__main__":
    PORT = 8000
    DASHBOARD_DIR.mkdir(exist_ok=True)
    server_address = ("", PORT)
    httpd = ThreadingHTTPServer(server_address, DashboardHandler)
    print(f"Starting dashboard server at http://localhost:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print("Server stopped.")
