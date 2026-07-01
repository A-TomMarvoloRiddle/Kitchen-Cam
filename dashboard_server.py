import json
import os
from http.server import SimpleHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Define paths
ROOT_DIR = Path(__file__).resolve().parent
LOGS_DIR = ROOT_DIR / "logs"
DASHBOARD_DIR = ROOT_DIR / "dashboard"

class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == "/api/runs":
            self.serve_api_runs()
        elif parsed_path.path == "/api/logs":
            qs = parse_qs(parsed_path.query)
            run_id = qs.get("run", [None])[0]
            self.serve_api_logs(run_id)
        else:
            super().do_GET()

    def serve_api_runs(self):
        """Return a list of available runs."""
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
        """Aggregate logs, optionally filtered by a specific run."""
        total_scans = 0
        total_violations = 0
        rule_breakdown = {
            "glove": {"status": "PASS", "detail": "Meets standard", "failed_count": 0},
            "hairnet": {"status": "PASS", "detail": "Meets standard", "failed_count": 0},
            "pest": {"status": "PASS", "detail": "Meets standard", "failed_count": 0}
        }
        
        # New: Detailed Chef-level analytics
        chef_stats = {}
        # New: Pest-level analytics
        pest_events = []

        if LOGS_DIR.exists():
            for log_file in LOGS_DIR.glob("*.json"):
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        
                        if not data:
                            continue
                            
                        # If filtering by run, check the first event's source
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
                                    
                                    # Collect pest analytics
                                    pest_events.append({
                                        "timestamp": event.get("timestamp"),
                                        "confidence": event.get("confidence", 0.0),
                                        "duration": event.get("duration_seconds", 0.0)
                                    })

                                # Track Chef level stats
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

        # Compute averages for chef stats
        for tid, stats in chef_stats.items():
            confs = stats["confidences"]
            stats["avg_confidence"] = round(sum(confs) / len(confs), 4) if confs else 0.0
            stats["labels"] = list(set(stats["labels"])) # Unique labels
            stats["duration_sum"] = round(stats["duration_sum"], 2)
            del stats["confidences"] # Clean up output

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
    httpd = HTTPServer(server_address, DashboardHandler)
    print(f"Starting dashboard server at http://localhost:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print("Server stopped.")
