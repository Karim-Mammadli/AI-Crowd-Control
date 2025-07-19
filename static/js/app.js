// AI Crowd Monitor - Enhanced Upload and Monitoring System

// Global state
let socket;
let isProcessing = false;
let modelsLoaded = false;
let currentMode = 'upload'; // 'upload', 'camera', 'video', 'image'
let currentProcessedFile = null;
let videoDetections = {}; // {frame_index: {timestamp, person_detections, face_detections}}

// Elements
let video, canvas, ctx;
let dropZone, imageUpload, videoUpload;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Initializing AI Crowd Monitor with seamless upload');
    initializeSocket();
    initializeUploadHandlers();
    initializeCanvas();
    setupUI();
});

function setupUI() {
    // Show placeholder initially
    showMediaContainer('placeholder');
    
    // Update initial status
    showStatus('🎯 Ready to analyze! Upload an image or video to start AI detection.', 'info');
    
    // Add event listeners for model selection
    const personModelSelect = document.getElementById('personModel');
    const faceModelSelect = document.getElementById('faceModel');
    
    if (personModelSelect) {
        personModelSelect.addEventListener('change', updateModelStatus);
    }
    
    if (faceModelSelect) {
        faceModelSelect.addEventListener('change', updateModelStatus);
    }
    
    // Update model status initially
    updateModelStatus();
    
    console.log('✅ UI setup complete');
}

function initializeSocket() {
    console.log('🔗 Connecting to WebSocket...');
    socket = io();
    
    // Socket event listeners
    socket.on('connect', function() {
        console.log('✅ Connected to server');
        showStatus('🔗 Connected to AI monitoring system', 'success');
    });
    
    socket.on('disconnect', function() {
        console.log('❌ Disconnected from server');
        showStatus('❌ Connection lost - please refresh the page', 'error');
    });
    
    socket.on('detection_update', function(data) {
        console.log('📊 Detection update received:', data);
        updateStats(data);
        addActivityLog(data);
    });
    
    socket.on('loading_progress', function(data) {
        console.log('📊 Model loading progress:', data);
        updateProgressBar(data);
    });
    
    socket.on('video_progress', function(data) {
        console.log('🎬 Video processing progress:', data);
        updateVideoProgress(data);
    });
    
    socket.on('video_processing_complete', function(data) {
        console.log('🎬 Video processing completed:', data);
        handleVideoProcessingComplete(data);
    });
    
    socket.on('processing_stopped', function(data) {
        console.log('⏹️ Processing stopped:', data);
        handleProcessingStopped(data);
    });
    
    socket.on('system_status', function(data) {
        console.log('⚙️ System status update:', data);
        handleSystemStatus(data);
    });
    
    socket.on('model_fallback', function(data) {
        console.log('⚠️ Model fallback detected:', data);
        showStatus(`⚠️ ${data.message}`, 'warning');
        
        // Update the model status to show actual model being used
        const statusElement = document.getElementById('modelStatus');
        if (statusElement) {
            let actualModelDisplay = data.actual_model.toUpperCase();
            if (data.actual_model === 'yolov11m') {
                actualModelDisplay = 'YOLOv11m (Latest)';
            } else if (data.actual_model === 'yolov8') {
                actualModelDisplay = 'YOLOv8';
            }
            
            statusElement.innerHTML = `🤖 Models: Ready | 👥 ${actualModelDisplay} (Fallback) + 👤 MediaPipe (Fast)`;
        }
    });
    
    socket.on('video_detection', function(data) {
        // Store detection by frame index
        videoDetections[data.frame_index] = data;
    });

    socket.on('live_frame', function(data) {
        const liveFeed = document.getElementById('liveFeed');
        if (liveFeed) {
            liveFeed.src = 'data:image/jpeg;base64,' + data.frame;
        }
    });
}

function initializeUploadHandlers() {
    console.log('📁 Setting up upload handlers...');
    
    // Get elements
    dropZone = document.getElementById('dropZone');
    imageUpload = document.getElementById('imageUpload');
    videoUpload = document.getElementById('videoUpload');
    
    // Image upload button click handler
    imageUpload.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            console.log('📷 Image file selected:', file.name);
            handleImageUpload(file);
        }
        // Reset input so same file can be selected again
        e.target.value = '';
    });
    
    // Video upload button click handler
    videoUpload.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            console.log('🎬 Video file selected:', file.name);
            handleVideoUpload(file);
        }
        // Reset input so same file can be selected again
        e.target.value = '';
    });
    
    // Drag and drop handlers
    dropZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    
    dropZone.addEventListener('dragleave', function(e) {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    });
    
    dropZone.addEventListener('drop', function(e) {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            console.log('📁 File dropped:', file.name, file.type);
            handleFileSelection(file);
        }
    });
    
    // Click on drop zone to select file
    dropZone.addEventListener('click', function() {
        console.log('📁 Drop zone clicked - opening file dialog');
        openFileDialog();
    });
    
    console.log('✅ Upload handlers initialized');
}

function openFileDialog() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.jpg,.jpeg,.png,.bmp,.webp,.mp4,.avi,.mov,.mkv,.webm';
    input.multiple = false;
    
    input.onchange = function(e) {
        const file = e.target.files[0];
        if (file) {
            console.log('📁 File selected from dialog:', file.name);
            handleFileSelection(file);
        }
    };
    
    input.click();
}

function handleFileSelection(file) {
    console.log('📁 Processing file selection:', file.name, 'Size:', formatFileSize(file.size));
    
    // Validate file size
    const maxSize = file.type.startsWith('video/') ? 100 * 1024 * 1024 : 10 * 1024 * 1024; // 100MB for video, 10MB for image
    if (file.size > maxSize) {
        const maxSizeText = file.type.startsWith('video/') ? '100MB' : '10MB';
        showStatus(`❌ File too large! Maximum size is ${maxSizeText}`, 'error');
        return;
    }
    
    // Determine file type and handle accordingly
    const fileType = file.type.toLowerCase();
    const fileName = file.name.toLowerCase();
    
    if (fileType.startsWith('image/') || 
        fileName.endsWith('.jpg') || fileName.endsWith('.jpeg') || 
        fileName.endsWith('.png') || fileName.endsWith('.bmp') || 
        fileName.endsWith('.webp')) {
        
        handleImageUpload(file);
        
    } else if (fileType.startsWith('video/') || 
               fileName.endsWith('.mp4') || fileName.endsWith('.avi') || 
               fileName.endsWith('.mov') || fileName.endsWith('.mkv') || 
               fileName.endsWith('.webm')) {
        
        handleVideoUpload(file);
        
    } else {
        showStatus('❌ Unsupported file format! Please use: Images (JPG, PNG, BMP, WEBP) or Videos (MP4, AVI, MOV, MKV, WEBM)', 'error');
    }
}

function handleImageUpload(file) {
    console.log('📷 Starting image upload and analysis:', file.name);
    
    // Get selected models
    const selectedModels = getSelectedModels();
    
    // Update UI immediately
    showStatus('📤 Uploading and analyzing image...', 'loading');
    setProcessingState(true);
    document.getElementById('mediaTitle').textContent = '📷 Analyzing Image...';
    
    // Show preview of original image while processing
    showImagePreview(file);
    
    // Create form data and upload
    const formData = new FormData();
    formData.append('file', file);
    formData.append('person_model', selectedModels.person_model);
    formData.append('face_model', selectedModels.face_model);
    
    fetch('/upload_image', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('📷 Image analysis complete:', data);
        
        if (data.success) {
            displayImageResults(data, file);
        } else {
            showStatus('❌ Image analysis failed: ' + data.message, 'error');
            setProcessingState(false);
        }
    })
    .catch(error => {
        console.error('❌ Image upload error:', error);
        showStatus('❌ Image upload failed: ' + error.message, 'error');
        setProcessingState(false);
    });
}

function handleVideoUpload(file) {
    console.log('🎬 Starting video upload and analysis:', file.name);
    
    // Get selected models
    const selectedModels = getSelectedModels();
    
    // Update UI immediately
    showStatus('📤 Uploading video for analysis...', 'loading');
    setProcessingState(true);
    document.getElementById('mediaTitle').textContent = '🎬 Uploading Video...';
    
    // Show video preview
    showVideoPreview(file);
    
    // Create form data and upload
    const formData = new FormData();
    formData.append('file', file);
    formData.append('person_model', selectedModels.person_model);
    formData.append('face_model', selectedModels.face_model);
    
    fetch('/upload_video', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('🎬 Video upload complete:', data);
        
        if (data.success) {
            // Start real-time video processing
            showStatus('🎬 Video uploaded! Starting AI analysis...', 'success');
            document.getElementById('mediaTitle').textContent = '🎬 Analyzing Video...';
            
            // Switch to live feed view
            document.getElementById('uploadedVideo').style.display = 'none';
            document.getElementById('overlay').style.display = 'none';
            document.getElementById('liveFeed').style.display = 'block';

            // Trigger server-side processing with model selection
            socket.emit('start_video_processing', {
                file_path: data.file_path,
                person_model: selectedModels.person_model,
                face_model: selectedModels.face_model
            });
            
        } else {
            showStatus('❌ Video upload failed: ' + data.message, 'error');
            setProcessingState(false);
        }
    })
    .catch(error => {
        console.error('❌ Video upload error:', error);
        showStatus('❌ Video upload failed: ' + error.message, 'error');
        setProcessingState(false);
    });
}

function showImagePreview(file) {
    currentMode = 'image';
    showMediaContainer('image');
    
    // Display original image immediately
    const originalImage = document.getElementById('originalImage');
    const processedImage = document.getElementById('processedImage');
    
    // Show original
    const reader = new FileReader();
    reader.onload = function(e) {
        originalImage.src = e.target.result;
        originalImage.style.opacity = '1';
    };
    reader.readAsDataURL(file);
    
    // Show loading placeholder for processed image
    processedImage.src = 'data:image/svg+xml;base64,' + btoa(`
        <svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 400 300">
            <rect width="400" height="300" fill="#333"/>
            <text x="200" y="150" text-anchor="middle" fill="white" font-size="16">
                Processing...
            </text>
        </svg>
    `);
    processedImage.style.opacity = '0.7';
}

function showVideoPreview(file) {
    currentMode = 'video';
    showMediaContainer('video');
    const video = document.getElementById('uploadedVideo');
    const url = URL.createObjectURL(file);
    video.src = url;
    video.load();
    // Setup canvas for overlay
    setupVideoCanvas();
    // Add timeupdate event for overlay sync
    video.addEventListener('timeupdate', function() {
        if (!ctx || !canvas) return;
        // Find the detection result with the closest timestamp to currentTime
        let closest = null;
        let minDiff = Infinity;
        for (let key in videoDetections) {
            let det = videoDetections[key];
            let diff = Math.abs(det.timestamp - video.currentTime);
            if (diff < minDiff) {
                minDiff = diff;
                closest = det;
            }
        }
        // Draw detections if found
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (closest) drawDetections(closest);
    });
}

function displayImageResults(data, originalFile) {
    console.log('🖼️ Displaying image analysis results');
    console.log('📊 Response data:', data);
    console.log('📁 Processed filename:', data.processed_filename);
    
    // Update processed image with results using file path
    const processedImage = document.getElementById('processedImage');
    // processedImage.src = 'data:image/jpeg;base64,' + data.processed_image; for base64 image
    const imagePath = '/processed/' + data.processed_filename;
    console.log('🖼️ Setting processed image src to:', imagePath);
    processedImage.src = imagePath;
    processedImage.style.opacity = '1';
    
    // Update stats immediately
    updateStats(data.stats);
    addActivityLog(data.stats);
    
    // Show download option
    showDownloadSection(true);
    // Use the processed_filename provided by the backend
    currentProcessedFile = data.processed_filename || data.processed_path.split('/').pop();
    
    // Update UI
    document.getElementById('mediaTitle').textContent = '📷 Image Analysis Complete';
    
    const peopleCount = data.stats.person_count;
    const faceCount = data.stats.face_count;
    const resultText = `Analysis complete! Found ${peopleCount} people and ${faceCount} faces.`;
    
    showStatus('✅ ' + resultText, 'success');
    setProcessingState(false);
    
    console.log('✅ Image analysis display complete');
}

function handleVideoProcessingComplete(data) {
    console.log('🎬 Video processing complete:', data);
    
    if (data.success) {
        const message = `Video analysis complete! Processed ${data.total_frames} frames.`;
        showStatus('✅ ' + message, 'success');
        document.getElementById('mediaTitle').textContent = '🎬 Video Analysis Complete';
        
        // Show download option
        showDownloadSection(true);
        currentProcessedFile = data.processed_path.split('/').pop();
        
        // Hide video progress bar
        const videoProgressContainer = document.getElementById('videoProgressContainer');
        videoProgressContainer.classList.remove('active');

        // Switch back from live feed to the final video player
        const liveFeed = document.getElementById('liveFeed');
        const uploadedVideo = document.getElementById('uploadedVideo');
        liveFeed.style.display = 'none';
        liveFeed.src = ''; // Clear the image to save memory
        uploadedVideo.style.display = 'block';
        
        // Load the processed video for playback
        uploadedVideo.src = `/download/${currentProcessedFile}`;
        uploadedVideo.load();
        
    } else {
        showStatus('❌ Video processing failed: ' + data.message, 'error');
    }
    
    setProcessingState(false);
}

function initializeCanvas() {
    console.log('🎨 Initializing canvas for detection overlay...');
    canvas = document.getElementById('overlay');
    if (canvas) {
        ctx = canvas.getContext('2d');
        ctx.lineWidth = 2;
        ctx.font = '14px Arial';
    }
    console.log('✅ Canvas initialized');
}

function setupVideoCanvas() {
    const video = document.getElementById('uploadedVideo');
    const canvas = document.getElementById('overlay');
    
    if (video && canvas) {
        video.addEventListener('loadedmetadata', function() {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            const rect = video.getBoundingClientRect();
            canvas.style.width = rect.width + 'px';
            canvas.style.height = rect.height + 'px';
            
            console.log('📐 Video canvas setup:', {
                videoWidth: video.videoWidth,
                videoHeight: video.videoHeight,
                displayWidth: rect.width,
                displayHeight: rect.height
            });
        });
    }
}

function showMediaContainer(type) {
    // Hide all containers first
    document.getElementById('imageContainer').style.display = 'none';
    document.getElementById('videoContainer').style.display = 'none';
    document.getElementById('cameraContainer').style.display = 'none';
    document.getElementById('placeholderContainer').style.display = 'none';
    
    // Show the requested container
    if (type === 'image') {
        document.getElementById('imageContainer').style.display = 'flex';
    } else if (type === 'video') {
        document.getElementById('videoContainer').style.display = 'block';
    } else if (type === 'camera') {
        document.getElementById('cameraContainer').style.display = 'block';
    } else {
        document.getElementById('placeholderContainer').style.display = 'flex';
    }
    
    console.log('📺 Media container switched to:', type);
}

function setProcessingState(processing) {
    isProcessing = processing;
    
    const stopBtn = document.getElementById('stopProcessingBtn');
    const imageBtn = document.getElementById('imageUploadBtn');
    const videoBtn = document.getElementById('videoUploadBtn');
    
    if (processing) {
        stopBtn.disabled = false;
        imageBtn.style.opacity = '0.6';
        videoBtn.style.opacity = '0.6';
        dropZone.style.opacity = '0.6';
        dropZone.style.pointerEvents = 'none';
    } else {
        stopBtn.disabled = true;
        imageBtn.style.opacity = '1';
        videoBtn.style.opacity = '1';
        dropZone.style.opacity = '1';
        dropZone.style.pointerEvents = 'auto';
    }
}

function stopProcessing() {
    console.log('⏹️ Stop processing requested by user');
    
    if (!isProcessing) {
        showStatus('ℹ️ No processing to stop', 'info');
        return;
    }
    
    socket.emit('stop_processing');
    showStatus('🛑 Processing stopped by user', 'info');
    setProcessingState(false);
    
    // Hide progress bars
    document.getElementById('progressContainer').classList.remove('active');
    document.getElementById('videoProgressContainer').classList.remove('active');
}

function handleProcessingStopped(data) {
    console.log('⏹️ Processing stopped by server:', data);
    setProcessingState(false);
    showStatus('⏹️ ' + (data.message || 'Processing stopped'), 'info');
}

function updateProgressBar(data) {
    console.log('📊 Model loading progress:', data.progress + '%');
    
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

function updateVideoProgress(data) {
    console.log('🎬 Video progress:', data.progress + '%', 'Frame:', data.frame);
    
    const videoProgressContainer = document.getElementById('videoProgressContainer');
    const videoProgressFill = document.getElementById('videoProgressFill');
    const videoProgressText = document.getElementById('videoProgressText');
    const videoProgressStep = document.getElementById('videoProgressStep');
    
    // Show progress bar
    videoProgressContainer.classList.add('active');
    
    // Update progress
    videoProgressFill.style.width = data.progress + '%';
    videoProgressText.textContent = data.progress + '%';
    videoProgressStep.textContent = data.message;
}

function handleSystemStatus(data) {
    const status = data.status;
    console.log('⚙️ System status:', status, '-', data.message);
    
    const modelStatus = document.getElementById('modelStatus');
    
    if (status === 'loading') {
        modelsLoaded = false;
        showStatus('🔄 ' + data.message, 'loading');
        modelStatus.innerHTML = '🔄 Loading: ' + data.message;
    } else if (status === 'ready') {
        modelsLoaded = true;
        showStatus('✅ ' + data.message, 'success');
        // Update model status with actual selected models
        updateModelStatus();
    } else if (status === 'error') {
        modelsLoaded = false;
        showStatus('❌ ' + data.message, 'error');
        modelStatus.innerHTML = '❌ Error: ' + data.message;
    }
}

function updateStats(data) {
    // Update statistics with smooth transitions
    const updates = {
        peopleCount: data.person_count || 0,
        faceCount: data.face_count || 0,
        crowdDensity: data.crowd_density || 'EMPTY',
        alertLevel: data.alert_level || 'NORMAL',
        systemStatus: data.system_status || 'Ready'
    };
    
    Object.keys(updates).forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            const newValue = String(updates[id]);
            if (element.textContent !== newValue) {
                element.textContent = newValue;
                // Add a subtle flash effect
                element.style.transition = 'all 0.3s ease';
                element.style.transform = 'scale(1.1)';
                setTimeout(() => {
                    element.style.transform = 'scale(1)';
                }, 300);
            }
        }
    });
    
    // Update alert level colors
    const alertElement = document.getElementById('alertLevel');
    if (alertElement) {
        alertElement.className = 'stat-value';
        if (data.alert_level === 'ALERT') {
            alertElement.classList.add('alert-high');
        } else if (data.alert_level === 'CAUTION') {
            alertElement.classList.add('alert-medium');
        } else {
            alertElement.classList.add('alert-normal');
        }
    }
}

function addActivityLog(data) {
    const logContainer = document.getElementById('activityLog');
    if (!logContainer) return;
    
    const time = new Date(data.timestamp || Date.now()).toLocaleTimeString();
    const activity = data.last_activity || 'Processing update';
    const newContent = `${time}: ${activity}`;
    
    // Only add if different from the last entry
    const lastItem = logContainer.firstElementChild;
    if (!lastItem || !lastItem.textContent.includes(activity)) {
        const logItem = document.createElement('div');
        logItem.className = 'activity-item';
        logItem.textContent = newContent;
        
        logContainer.insertBefore(logItem, logContainer.firstChild);
        
        // Keep only last 15 entries
        while (logContainer.children.length > 15) {
            logContainer.removeChild(logContainer.lastChild);
        }
        
        console.log('📝 Activity logged:', activity);
    }
}

function drawDetections(data) {
    if (!ctx || !canvas || currentMode !== 'video') return;
    
    const video = document.getElementById('uploadedVideo');
    if (!video || !video.videoWidth) return;
    
    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Get scaling factors
    const scaleX = canvas.width / video.videoWidth;
    const scaleY = canvas.height / video.videoHeight;
    
    // Draw person detections (green boxes)
    if (data.person_detections && data.person_detections.length > 0) {
        ctx.strokeStyle = '#4CAF50';
        ctx.fillStyle = '#4CAF50';
        ctx.lineWidth = 3;
        
        data.person_detections.forEach((detection, index) => {
            const [x1, y1, x2, y2] = detection.bbox;
            const x = x1 * scaleX;
            const y = y1 * scaleY;
            const width = (x2 - x1) * scaleX;
            const height = (y2 - y1) * scaleY;
            
            // Draw bounding box
            ctx.strokeRect(x, y, width, height);
            
            // Draw label with background
            const label = `Person ${index + 1}: ${(detection.confidence * 100).toFixed(0)}%`;
            const textWidth = ctx.measureText(label).width;
            
            // Background for text
            ctx.fillRect(x, y - 30, textWidth + 12, 25);
            
            // Text
            ctx.fillStyle = 'white';
            ctx.fillText(label, x + 6, y - 10);
            ctx.fillStyle = '#4CAF50';
        });
    }
    
    // Draw face detections (blue boxes)
    if (data.face_detections && data.face_detections.length > 0) {
        ctx.strokeStyle = '#2196F3';
        ctx.fillStyle = '#2196F3';
        ctx.lineWidth = 2;
        
        data.face_detections.forEach((detection, index) => {
            const [x1, y1, x2, y2] = detection.bbox;
            const x = x1 * scaleX;
            const y = y1 * scaleY;
            const width = (x2 - x1) * scaleX;
            const height = (y2 - y1) * scaleY;
            
            // Draw bounding box
            ctx.strokeRect(x, y, width, height);
            
            // Draw label with background
            const label = `Face ${index + 1}: ${(detection.confidence * 100).toFixed(0)}%`;
            const textWidth = ctx.measureText(label).width;
            
            // Background for text
            ctx.fillRect(x, y - 30, textWidth + 12, 25);
            
            // Text
            ctx.fillStyle = 'white';
            ctx.fillText(label, x + 6, y - 10);
            ctx.fillStyle = '#2196F3';
        });
    }
}

function showDownloadSection(show) {
    const downloadSection = document.getElementById('downloadSection');
    if (downloadSection) {
        downloadSection.style.display = show ? 'block' : 'none';
    }
}

function downloadResults() {
    if (currentProcessedFile) {
        const filename = currentProcessedFile;
        const downloadUrl = `/download/${filename}`;
        console.log('💾 Downloading processed file:', downloadUrl);
        
        // Create download link and trigger
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        showStatus('📥 Download started!', 'success');
    } else {
        showStatus('❌ No processed file available for download', 'error');
    }
}

function showStatus(message, type) {
    const statusDiv = document.getElementById('status');
    if (statusDiv) {
        statusDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
        
        console.log(`📢 Status: [${type.toUpperCase()}] ${message}`);
        
        // Auto-hide after delay (except loading states)
        if (type !== 'loading') {
            setTimeout(() => {
                if (statusDiv.innerHTML.includes(message)) {
                    statusDiv.innerHTML = '';
                }
            }, 5000);
        }
    }
}

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Global functions (called from HTML)
window.startMonitoring = function() {
    showStatus('ℹ️ Camera mode is disabled. Please use file upload instead.', 'info');
};

window.stopMonitoring = stopProcessing;
window.downloadResults = downloadResults;

// Initialize when DOM is ready
console.log('📋 Enhanced upload system ready!');

// Model selection functions
function getSelectedModels() {
    const personModel = document.getElementById('personModel').value;
    const faceModel = document.getElementById('faceModel').value;
    
    console.log(`🤖 Selected models - Person: ${personModel}, Face: ${faceModel}`);
    
    return {
        person_model: personModel,
        face_model: faceModel
    };
}

function updateModelStatus() {
    const models = getSelectedModels();
    const statusElement = document.getElementById('modelStatus');
    
    if (statusElement) {
        let personModelDisplay = models.person_model.toUpperCase();
        if (models.person_model === 'yolov11m') {
            personModelDisplay = 'YOLOv11m (Latest)';
        } else if (models.person_model === 'yolov8') {
            personModelDisplay = 'YOLOv8';
        }
        
        let faceModelDisplay = models.face_model.toUpperCase();
        if (models.face_model === 'mediapipe') {
            faceModelDisplay = 'MediaPipe (Fast)';
        }
        
        const statusText = modelsLoaded ? 'Ready' : 'Ready to load';
        statusElement.innerHTML = `🤖 Models: ${statusText} | 👥 ${personModelDisplay} + 👤 ${faceModelDisplay}`;
    }
}