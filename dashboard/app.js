document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initUploadLogic();
    fetchRuns();
    fetchLogs(); // Initial fetch for dashboard (all runs)
});

let violationsChart = null;
let currentRun = "";

function initTabs() {
    const tabUpload = document.getElementById("tab-upload");
    const tabMonitoring = document.getElementById("tab-monitoring");
    const tabDashboard = document.getElementById("tab-dashboard");
    const viewUpload = document.getElementById("view-upload");
    const viewMonitoring = document.getElementById("view-monitoring");
    const viewDashboard = document.getElementById("view-dashboard");

    function resetTabs() {
        tabUpload.classList.remove("active");
        tabMonitoring.classList.remove("active");
        tabDashboard.classList.remove("active");
        viewUpload.style.display = "none";
        viewMonitoring.style.display = "none";
        viewDashboard.style.display = "none";
    }

    tabUpload.addEventListener("click", (e) => {
        e.preventDefault();
        resetTabs();
        tabUpload.classList.add("active");
        viewUpload.style.display = "block";
    });

    tabMonitoring.addEventListener("click", (e) => {
        e.preventDefault();
        resetTabs();
        tabMonitoring.classList.add("active");
        viewMonitoring.style.display = "block";
    });

    tabDashboard.addEventListener("click", (e) => {
        e.preventDefault();
        resetTabs();
        tabDashboard.classList.add("active");
        viewDashboard.style.display = "block";
    });

    // Run selector listener
    document.getElementById("run-selector").addEventListener("change", (e) => {
        currentRun = e.target.value;
        fetchLogs(currentRun);
    });
}

function initUploadLogic() {
    const fileInput = document.getElementById("video-file");
    const btnSelect = document.getElementById("btn-select-file");
    const btnUpload = document.getElementById("btn-upload");
    const fileNameDisplay = document.getElementById("selected-file-name");
    const progressContainer = document.getElementById("progress-container");
    const progressFill = document.getElementById("upload-progress");
    const progressText = document.getElementById("upload-status-text");
    const streamContainer = document.getElementById("stream-container");
    const liveStream = document.getElementById("live-stream");
    const streamPlaceholder = document.getElementById("stream-placeholder");

    let selectedFile = null;
    let pollInterval = null;

    btnSelect.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            selectedFile = e.target.files[0];
            fileNameDisplay.innerText = selectedFile.name;
            btnUpload.disabled = false;
            
            // Reset the stream box in case of a previous run
            progressContainer.style.display = "none";
            progressFill.style.width = "0%";
            progressText.innerText = "Uploading: 0%";
            liveStream.style.display = "none";
            streamPlaceholder.style.display = "block";
            streamPlaceholder.innerText = "Stream will appear here once processing starts.";
        }
    });

    btnUpload.addEventListener("click", () => {
        if (!selectedFile) return;
        
        btnUpload.disabled = true;
        btnSelect.disabled = true;
        progressContainer.style.display = "block";
        progressFill.style.width = "50%";
        progressText.innerText = "Uploading to server...";

        // Simple POST fetch for binary file data
        fetch(`/api/upload?filename=${encodeURIComponent(selectedFile.name)}`, {
            method: 'POST',
            body: selectedFile,
            headers: {
                'Content-Type': 'application/octet-stream'
            }
        })
        .then(res => res.json())
        .then(data => {
            progressFill.style.width = "100%";
            progressText.innerText = "Processing video...";
            
            // Show MJPEG stream
            streamPlaceholder.style.display = "none";
            liveStream.style.display = "block";
            // Append timestamp to bypass caching
            liveStream.src = `/api/stream?t=${new Date().getTime()}`;
            
            // Start polling for status
            pollInterval = setInterval(checkStatus, 2000);
        })
        .catch(err => {
            console.error("Upload failed", err);
            progressText.innerText = "Upload failed!";
            btnUpload.disabled = false;
            btnSelect.disabled = false;
        });
    });

    function checkStatus() {
        fetch('/api/status')
            .then(res => res.json())
            .then(data => {
                if (data.status === "DONE") {
                    clearInterval(pollInterval);
                    progressText.innerText = "Processing Complete!";
                    liveStream.style.display = "none";
                    streamPlaceholder.style.display = "block";
                    streamPlaceholder.innerText = "Processing finished. Redirecting...";
                    
                    // Fetch runs again to populate the new run
                    fetchRuns().then(() => {
                        // Switch to monitoring tab
                        document.getElementById("tab-monitoring").click();
                        // Select the new run
                        const selector = document.getElementById("run-selector");
                        selector.value = data.run_id;
                        currentRun = data.run_id;
                        fetchLogs(currentRun);
                        
                        // Reset upload UI
                        btnUpload.disabled = false;
                        btnSelect.disabled = false;
                        progressContainer.style.display = "none";
                        fileNameDisplay.innerText = "No file selected";
                        selectedFile = null;
                        fileInput.value = "";
                    });
                }
            })
            .catch(console.error);
    }
}

async function fetchRuns() {
    try {
        const response = await fetch('/api/runs');
        const data = await response.json();
        
        const selector = document.getElementById("run-selector");
        selector.innerHTML = ''; // Removed All Runs (Aggregated)
        
        if (data.runs.length > 0) {
            data.runs.forEach(run => {
                const opt = document.createElement("option");
                opt.value = run;
                opt.innerText = `Run: ${run}`;
                selector.appendChild(opt);
            });
            
            // Set current run to first option and fetch its logs initially if in monitoring view
            if (!currentRun) {
                currentRun = data.runs[0];
            }
        } else {
            selector.innerHTML = '<option value="">No runs available</option>';
        }
    } catch (error) {
        console.error("Failed to fetch runs:", error);
    }
}

async function fetchLogs(runId = "") {
    try {
        let url = '/api/logs';
        if (runId) {
            url += `?run=${encodeURIComponent(runId)}`;
        }
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const data = await response.json();
        updateDashboard(data);
        updateMonitoringView(data);
    } catch (error) {
        console.error("Failed to fetch logs:", error);
        document.getElementById("compliance-score").innerText = "ERR";
        document.getElementById("compliance-status").innerText = "Error";
        document.getElementById("compliance-status").classList.add("fail");
    }
}

function updateDashboard(data) {
    // Update Score Card
    const scoreEl = document.getElementById("compliance-score");
    const statusEl = document.getElementById("compliance-status");
    
    scoreEl.innerText = data.compliance_score;
    if (data.compliance_score < 100) {
        scoreEl.classList.add("fail");
        statusEl.innerText = "Non-Compliant";
        statusEl.classList.add("fail");
    } else {
        scoreEl.classList.remove("fail");
        statusEl.innerText = "Compliant";
        statusEl.classList.remove("fail");
    }

    // Update Stats
    document.getElementById("rules-passed").innerText = data.rules_passed;
    document.getElementById("rules-passed-text").innerText = data.rules_passed;
    document.getElementById("rules-failed").innerText = data.rules_failed;
    document.getElementById("rules-failed-text").innerText = data.rules_failed;
    document.getElementById("total-violations").innerText = `${data.total_violations} events`;

    // Update Rule Table
    const tbody = document.getElementById("rule-table-body");
    tbody.innerHTML = "";

    const ruleNames = {
        "glove": "Glove Compliance",
        "hairnet": "Hairnet Protocol",
        "pest": "Pest Control"
    };

    const chartLabels = [];
    const chartData = [];

    for (const [ruleKey, ruleInfo] of Object.entries(data.rule_breakdown)) {
        const tr = document.createElement("tr");
        const tdRule = document.createElement("td");
        tdRule.innerText = ruleNames[ruleKey] || ruleKey;
        
        const tdStatus = document.createElement("td");
        const badge = document.createElement("span");
        badge.className = `badge ${ruleInfo.status === 'PASS' ? 'badge-pass' : 'badge-fail'}`;
        badge.innerText = ruleInfo.status;
        tdStatus.appendChild(badge);
        
        const tdDetail = document.createElement("td");
        tdDetail.innerText = ruleInfo.detail;
        if (ruleInfo.status === 'FAIL') tdDetail.style.color = 'var(--danger-color)';

        tr.appendChild(tdRule);
        tr.appendChild(tdStatus);
        tr.appendChild(tdDetail);
        tbody.appendChild(tr);

        chartLabels.push(ruleNames[ruleKey] || ruleKey);
        chartData.push(ruleInfo.failed_count);
    }

    renderChart(chartLabels, chartData);
}

function updateMonitoringView(data) {
    const tbody = document.getElementById("chef-table-body");
    tbody.innerHTML = "";
    
    if (!data.chef_analytics || data.chef_analytics.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; padding: 2rem;">No chef-level violations found for this run.</td></tr>';
        return;
    }

    // Sort by track ID ascending
    const sortedChefs = data.chef_analytics.sort((a, b) => a.track_id - b.track_id);

    sortedChefs.forEach(chef => {
        const tr = document.createElement("tr");
        
        // ID
        const tdId = document.createElement("td");
        tdId.innerHTML = `<strong>#${chef.track_id}</strong>`;
        
        // Violations Count
        const tdCount = document.createElement("td");
        tdCount.innerText = chef.violations;
        
        // Duration
        const tdDuration = document.createElement("td");
        tdDuration.innerText = chef.duration_sum.toFixed(2);
        
        // Avg Confidence
        const tdConf = document.createElement("td");
        tdConf.innerText = (chef.avg_confidence * 100).toFixed(1) + "%";
        
        // Labels Failed
        const tdLabels = document.createElement("td");
        chef.labels.forEach(label => {
            const span = document.createElement("span");
            span.className = "label-badge";
            span.innerText = label.replace("missing_", "").toUpperCase();
            tdLabels.appendChild(span);
        });

        tr.appendChild(tdId);
        tr.appendChild(tdCount);
        tr.appendChild(tdDuration);
        tr.appendChild(tdConf);
        tr.appendChild(tdLabels);
        tbody.appendChild(tr);
    });

    // Populate Pest Table
    const pestBody = document.getElementById("pest-table-body");
    pestBody.innerHTML = "";
    
    if (!data.pest_analytics || data.pest_analytics.length === 0) {
        pestBody.innerHTML = '<tr><td colspan="4" style="text-align: center; padding: 2rem;">No pest detections found for this run.</td></tr>';
    } else {
        data.pest_analytics.forEach(pest => {
            const tr = document.createElement("tr");
            
            // Timestamp
            const tdTime = document.createElement("td");
            const date = new Date(pest.timestamp);
            tdTime.innerText = date.toLocaleString();
            
            // Confidence
            const tdConf = document.createElement("td");
            tdConf.innerText = (pest.confidence * 100).toFixed(1) + "%";
            
            // Duration
            const tdDur = document.createElement("td");
            tdDur.innerText = pest.duration.toFixed(2);
            
            // Status
            const tdStatus = document.createElement("td");
            const badge = document.createElement("span");
            badge.className = "label-badge";
            badge.innerText = "DETECTED";
            tdStatus.appendChild(badge);
            
            tr.appendChild(tdTime);
            tr.appendChild(tdConf);
            tr.appendChild(tdDur);
            tr.appendChild(tdStatus);
            pestBody.appendChild(tr);
        });
    }
}

function renderChart(labels, data) {
    const ctx = document.getElementById('violationsChart').getContext('2d');
    if (violationsChart) violationsChart.destroy();

    const gradient = ctx.createLinearGradient(0, 0, 400, 0);
    gradient.addColorStop(0, 'rgba(239, 68, 68, 0.8)');
    gradient.addColorStop(1, 'rgba(239, 68, 68, 1)');

    violationsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Violation Events',
                data: data,
                backgroundColor: gradient,
                borderRadius: 4,
                borderSkipped: false,
                barThickness: 24
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    beginAtZero: true,
                    grid: { color: '#f3f4f6', drawBorder: false },
                    ticks: { color: '#6b7280', stepSize: 1 }
                },
                y: {
                    grid: { display: false, drawBorder: false },
                    ticks: { font: { family: 'Inter', weight: 500 }, color: '#4b5563' }
                }
            },
            animation: { duration: 1000, easing: 'easeOutQuart' }
        }
    });
}
