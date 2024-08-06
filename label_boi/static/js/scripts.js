document.addEventListener('DOMContentLoaded', function() {
    const uploadSection = document.getElementById('uploadSection');
    const configSection = document.getElementById('configSection');
    const cropSection = document.getElementById('cropSection');
    const dropArea = document.getElementById('dropArea');
    const videoInput = document.getElementById('videoInput');
    const uploadButton = document.getElementById('uploadButton');
    const fragmentButton = document.getElementById('fragmentButton');
    const previousButton = document.getElementById('previousButton');
    const nextButton = document.getElementById('nextButton');
    const cropButton = document.getElementById('cropButton');
    const image = document.getElementById('image');
    const cropArea = document.getElementById('cropArea');
    const statusMessage = document.getElementById('statusMessage');

    let frames = [];
    let currentFrameIndex = 0;
    let currentVideoFile = null;
    let videoDuration = 0;

    function formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    // Drag and drop functionality
    dropArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropArea.style.backgroundColor = '#e9e9e9';
    });

    dropArea.addEventListener('dragleave', () => {
        dropArea.style.backgroundColor = '';
    });

    dropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        dropArea.style.backgroundColor = '';
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('video/')) {
            handleVideoUpload(file);
        }
    });

    uploadButton.addEventListener('click', () => videoInput.click());

    videoInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleVideoUpload(file);
        }
    });

    function handleVideoUpload(file) {
        const formData = new FormData();
        formData.append('file', file);

        fetch('/upload_video', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                statusMessage.textContent = data.error;
            } else {
                currentVideoFile = data.filename;
                videoDuration = data.duration;
                initializeTimeSlider();
                uploadSection.style.display = 'none';
                configSection.style.display = 'block';
            }
        })
        .catch(error => {
            statusMessage.textContent = 'Error uploading video: ' + error;
        });
    }

    function initializeTimeSlider() {
        $("#timeSlider").slider({
            range: true,
            min: 0,
            max: videoDuration,
            values: [0, videoDuration],
            slide: function(event, ui) {
                $("#startTimeValue").text(formatTime(ui.values[0]));
                $("#endTimeValue").text(formatTime(ui.values[1]));
            }
        });
        $("#startTimeValue").text(formatTime(0));
        $("#endTimeValue").text(formatTime(videoDuration));
    }

    fragmentButton.addEventListener('click', () => {
        const interval = document.getElementById('interval').value;
        const startTime = $("#timeSlider").slider("values", 0);
        const endTime = $("#timeSlider").slider("values", 1);

        fetch('/fragment_video', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                filename: currentVideoFile,
                interval: interval,
                start_time: startTime,
                end_time: endTime
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                statusMessage.textContent = data.error;
            } else {
                frames = data.frames;
                configSection.style.display = 'none';
                cropSection.style.display = 'block';
                loadFrame(0);
            }
        })
        .catch(error => {
            statusMessage.textContent = 'Error processing video: ' + error;
        });
    });

    function loadFrame(index) {
        if (index >= 0 && index < frames.length) {
            currentFrameIndex = index;
            image.src = '/frames/' + frames[index];
            // Reset crop area
            cropArea.style.width = '100px';
            cropArea.style.height = '100px';
            cropArea.style.left = '0px';
            cropArea.style.top = '0px';
        }
    }

    previousButton.addEventListener('click', () => {
        if (currentFrameIndex > 0) {
            loadFrame(currentFrameIndex - 1);
        }
    });

    nextButton.addEventListener('click', () => {
        if (currentFrameIndex < frames.length - 1) {
            loadFrame(currentFrameIndex + 1);
        }
    });

    cropButton.addEventListener('click', () => {
        const rect = cropArea.getBoundingClientRect();
        const imageRect = image.getBoundingClientRect();

        fetch('/crop', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                filename: frames[currentFrameIndex],
                x: rect.left - imageRect.left,
                y: rect.top - imageRect.top,
                width: rect.width,
                height: rect.height
            }),
        })
        .then(response => response.json())
        .then(data => {
            statusMessage.textContent = data.message;
            if (currentFrameIndex < frames.length - 1) {
                loadFrame(currentFrameIndex + 1);
            }
        })
        .catch(error => {
            statusMessage.textContent = 'Error cropping image: ' + error;
        });
    });

    // Make crop area draggable and resizable
    let isDragging = false;
    let isResizing = false;
    let startX, startY, startWidth, startHeight;

    cropArea.addEventListener('mousedown', (e) => {
        if (e.target === cropArea) {
            isDragging = true;
        } else {
            isResizing = true;
        }
        startX = e.clientX - cropArea.offsetLeft;
        startY = e.clientY - cropArea.offsetTop;
        startWidth = parseInt(getComputedStyle(cropArea).width, 10);
        startHeight = parseInt(getComputedStyle(cropArea).height, 10);
    });

    document.addEventListener('mousemove', (e) => {
        if (isDragging) {
            const newLeft = e.clientX - startX;
            const newTop = e.clientY - startY;
            cropArea.style.left = `${newLeft}px`;
            cropArea.style.top = `${newTop}px`;
        } else if (isResizing) {
            const newWidth = startWidth + (e.clientX - (startX + cropArea.offsetLeft));
            const newHeight = startHeight + (e.clientY - (startY + cropArea.offsetTop));
            cropArea.style.width = `${newWidth}px`;
            cropArea.style.height = `${newHeight}px`;
        }
    });

    document.addEventListener('mouseup', () => {
        isDragging = false;
        isResizing = false;
    });
});