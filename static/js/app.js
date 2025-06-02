 // AI Crowd Monitor - Frontend JavaScript

// Global state
let socket;
let isMonitoring = false;
let isInitializing = false;
let modelsLoaded = false;

// Canvas elements for drawing detections
let video, canvas, ctx;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Initializing AI Crowd Monitor frontend');
    initializeSocket();
    initializeWebcam();
    initializeCanvas();
});

function initializeSocket() {
    console.log('🔗 Connecting to WebSocket...');
    socket = io();
    
    // Socket event listeners
    socket.on('connect', function() {
        console.log('✅ Connected to server');
        showStatus('🔗 Connected to monitoring system', 'success');
    });
    
    socket.on('disconnect', function() {
        console.log('❌ Disconnected from server');
        showStatus('❌ Disconnected from server', 'error');
        updateMonitoringStatus(false);
    });
    
    socket.on('detection_update', function(data) {
        updateStats(data);
        addActivityLog(data);
        drawDetections(data);
    });
    
    socket.on('monitoring_status', function(data) {
        console.log('📊 Monitoring status update:', data);
        updateMonitoringStatus(data.active);
        showStatus(data.message, data.active ? 'success' : 'info');
    });
    
    socket.on('system_status', function(data) {
        handleSystemStatus(data);
    });
    
    socket.on('loading_progress', function(data) {
        updateProgressBar(data);
    });
    
    // Debug: Log all socket events
    socket.onAny((event, ...args) => {
        console.log('Socket event:', event, args);
    });
}

function initializeWebcam() {
    console.log('📹 Initializing webcam...');
    video = document.getElementById('video');
    
    navigator.mediaDevices.getUserMedia({ 
        video: { 
            width: { ideal: 1280 }, 
            height: { ideal: 720 } 
        } 
    })
    .then(stream => {
        video.srcObject = stream;
        video.onloadedmetadata = function() {
            setupCanvas();
        };
        console.log('✅ Webcam initialized successfully');
        showStatus('📹 Camera connected successfully', 'success');
    })
    .catch(err => {
        console.error('❌ Camera error:', err);
        showStatus('❌ Camera access denied: ' + err.message, 'error');
    });
}

function initializeCanvas() {
    console.log('🎨 Initializing canvas overlay...');
    canvas = document.getElementById('overlay');
    ctx = canvas.getContext('2d');
    
    // Setup canvas properties
    ctx.lineWidth = 2;
    ctx.font = '14px Arial';
    
    console.log('✅ Canvas initialized');
}

function setupCanvas() {
    if (video && canvas) {
        // Match canvas size to video display size
        const rect = video.getBoundingClientRect();
        canvas.width = video.videoWidth || rect.width;
        canvas.height = video.videoHeight || rect.height;
        canvas.style.width = rect.width + 'px';
        canvas.style.height = rect.height + 'px';
        
        console.log('📐 Canvas setup:', {
            width: canvas.width,
            height: canvas.height,
            displayWidth: rect.width,
            displayHeight: rect.height
        });
    }
}

// Window resize handler
window.addEventListener('resize', setupCanvas);

// Main control functions
function startMonitoring() {
    console.log('▶️ Start button clicked. Current state:', {isMonitoring, isInitializing, modelsLoaded});
    
    if (isMonitoring || isInitializing) {
        console.log('⚠️ Ignoring start request - already active or initializing');
        return;
    }
    
    const startBtn = document.getElementById('startBtn');
    startBtn.disabled = true;
    startBtn.innerHTML = '<span class="loading-spinner"></span>Starting...';
    
    console.log('📡 Emitting start_monitoring event');
    socket.emit('start_monitoring');
}

function stopMonitoring() {
    console.log('⏹️ Stop button clicked. Current state:', {isMonitoring});
    
    if (!isMonitoring) {
        console.log('⚠️ Ignoring stop request - not monitoring');
        return;
    }
    
    const stopBtn = document.getElementById('stopBtn');
    stopBtn.disabled = true;
    stopBtn.textContent = 'Stopping...';
    
    // Clear canvas
    if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    
    console.log('📡 Emitting stop_monitoring event');
    socket.emit('stop_monitoring');
}

function updateProgressBar(data) {
    console.log('📊 Progress update:', data);
    
    const progressContainer = document.getElementById('progressContainer');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const progressStep = document.getElementById('progressStep');
    
    // Show progress bar
    progressContainer.classList.add('active');
    
    // Update progress
    progressFill.style.width = data.progress + '%';
    progressText.textContent = data.progress + '%';
    progressStep.textContent = data.message;
    
    // Hide when complete
    if (data.progress >= 100) {
        setTimeout(() => {
            progressContainer.classList.remove('active');
        }, 2000);
    }
}

function handleSystemStatus(data) {
    const status = data.status;
    console.log('⚙️ System status:', data);
    
    const modelStatus = document.getElementById('modelStatus');
    const startBtn = document.getElementById('startBtn');
    
    if (status === 'loading') {
        isInitializing = true;
        modelsLoaded = false;
        showStatus(data.message, 'loading');
        startBtn.innerHTML = '<span class="loading-spinner"></span>Loading AI Models...';
        modelStatus.innerHTML = '🔄 Loading: ' + data.message;
    } else if (status === 'ready') {
        isInitializing = false;
        modelsLoaded = true;
        showStatus(data.message, 'success');
        modelStatus.innerHTML = '✅ Models loaded: YOLOv8 + MediaPipe ready';
        // Don't reset button text here - let monitoring_status handle it
    } else if (status === 'error') {
        isInitializing = false;
        modelsLoaded = false;
        showStatus(data.message, 'error');
        startBtn.textContent = 'Start Monitoring';
        startBtn.disabled = false;
        modelStatus.innerHTML = '❌ Error: ' + data.message;
    }
}

function updateStats(data) {
    // Update all statistics smoothly
    const updates = {
        peopleCount: data.person_count,
        faceCount: data.face_count,
        crowdDensity: data.crowd_density,
        alertLevel: data.alert_level,
        systemStatus: data.system_status
    };
    
    Object.keys(updates).forEach(id => {
        const element = document.getElementById(id);
        if (element && element.textContent !== String(updates[id])) {
            element.textContent = updates[id];
        }
    });
    
    // Update alert level colors
    const alertElement = document.getElementById('alertLevel');
    alertElement.className = 'stat-value';
    if (data.alert_level === 'ALERT') {
        alertElement.classList.add('alert-high');
    } else if (data.alert_level === 'CAUTION') {
        alertElement.classList.add('alert-medium');
    } else {
        alertElement.classList.add('alert-normal');
    }
}

function addActivityLog(data) {
    const logContainer = document.getElementById('activityLog');
    const time = new Date(data.timestamp).toLocaleTimeString();
    
    // Create new log entry
    const newContent = `${time}: ${data.last_activity}`;
    
    // Only add if it's different from the last entry
    const lastItem = logContainer.firstElementChild;
    if (!lastItem || !lastItem.textContent.includes(data.last_activity)) {
        const logItem = document.createElement('div');
        logItem.className = 'activity-item';
        logItem.textContent = newContent;
        
        logContainer.insertBefore(logItem, logContainer.firstChild);
        
        // Keep only last 12 entries
        while (logContainer.children.length > 12) {
            logContainer.removeChild(logContainer.lastChild);
        }
    }
}

function drawDetections(data) {
    if (!ctx || !canvas) return;
    
    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Get scaling factors
    const scaleX = canvas.width / (video.videoWidth || canvas.width);
    const scaleY = canvas.height / (video.videoHeight || canvas.height);
    
    // Draw person detections (green boxes)
    if (data.person_detections && data.person_detections.length > 0) {
        ctx.strokeStyle = '#4CAF50'; // Green
        ctx.fillStyle = '#4CAF50';
        
        data.person_detections.forEach(detection => {
            const [x1, y1, x2, y2] = detection.bbox;
            const x = x1 * scaleX;
            const y = y1 * scaleY;
            const width = (x2 - x1) * scaleX;
            const height = (y2 - y1) * scaleY;
            
            // Draw bounding box
            ctx.strokeRect(x, y, width, height);
            
            // Draw label
            const label = `Person: ${(detection.confidence * 100).toFixed(0)}%`;
            const textWidth = ctx.measureText(label).width;
            
            // Background for text
            ctx.fillRect(x, y - 25, textWidth + 10, 20);
            
            // Text
            ctx.fillStyle = 'white';
            ctx.fillText(label, x + 5, y - 10);
            ctx.fillStyle = '#4CAF50';
        });
    }
    
    // Draw face detections (blue boxes)
    if (data.face_detections && data.face_detections.length > 0) {
        ctx.strokeStyle = '#2196F3'; // Blue
        ctx.fillStyle = '#2196F3';
        
        data.face_detections.forEach(detection => {
            const [x1, y1, x2, y2] = detection.bbox;
            const x = x1 * scaleX;
            const y = y1 * scaleY;
            const width = (x2 - x1) * scaleX;
            const height = (y2 - y1) * scaleY;
            
            // Draw bounding box
            ctx.strokeRect(x, y, width, height);
            
            // Draw label
            const label = `Face: ${(detection.confidence * 100).toFixed(0)}%`;
            const textWidth = ctx.measureText(label).width;
            
            // Background for text
            ctx.fillRect(x, y - 25, textWidth + 10, 20);
            
            // Text
            ctx.fillStyle = 'white';
            ctx.fillText(label, x + 5, y - 10);
            ctx.fillStyle = '#2196F3';
        });
    }
}

function updateMonitoringStatus(active) {
    console.log('🔄 Updating monitoring status:', active);
    isMonitoring = active;
    
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    
    if (active) {
        // Monitoring is active
        startBtn.disabled = true;
        startBtn.textContent = 'Monitoring Active';
        stopBtn.disabled = false;
        stopBtn.textContent = 'Stop Monitoring';
    } else {
        // Monitoring is stopped
        startBtn.disabled = isInitializing;
        startBtn.textContent = modelsLoaded ? 'Start Monitoring' : 'Start Monitoring';
        stopBtn.disabled = true;
        stopBtn.textContent = 'Stopped';
    }
}

function showStatus(message, type) {
    const statusDiv = document.getElementById('status');
    statusDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
    
    // Auto-hide after 4 seconds (except loading states)
    if (type !== 'loading') {
        setTimeout(() => {
            if (statusDiv.innerHTML.includes(message)) {
                statusDiv.innerHTML = '';
            }
        }, 4000);
    }
}

// Prevent accidental page refresh during monitoring
window.addEventListener('beforeunload', function(e) {
    if (isMonitoring) {
        e.preventDefault();
        e.returnValue = 'Monitoring is active. Are you sure you want to leave?';
        return e.returnValue;
    }
});

// Utility functions
function formatTimestamp(timestamp) {
    return new Date(timestamp).toLocaleTimeString();
}

function logDebug(message) {
    console.log(`🐛 [DEBUG] ${message}`);
}

function logError(message, error = null) {
    console.error(`❌ [ERROR] ${message}`, error);
}

// Initialize everything when DOM is ready
console.log('📋 Frontend script loaded successfully');