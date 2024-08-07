document.addEventListener('DOMContentLoaded', function() {
    const uploadSection = document.getElementById('uploadSection');
    const configSection = document.getElementById('configSection');
    const cropSection = document.getElementById('cropSection');
    const segmentationSection = document.getElementById('segmentationSection');
    const dropArea = document.getElementById('dropArea');
    const videoInput = document.getElementById('videoInput');
    const uploadButton = document.getElementById('uploadButton');
    const fragmentButton = document.getElementById('fragmentButton');
    const previousButton = document.getElementById('previousButton');
    const nextButton = document.getElementById('nextButton');
    const cropButton = document.getElementById('cropButton');
    const image = document.getElementById('image');
    const imageSegmentation = document.getElementById('imageSegmentation');
    const cropArea = document.getElementById('cropArea');
    const statusMessage = document.getElementById('statusMessage');
    const cropCountElement = document.getElementById('cropCount');
    const segmentationCanvas = document.getElementById('segmentationCanvas');
    const applySegmentationButton = document.getElementById('applySegmentationButton');
    const previousButtonSegmentation = document.getElementById('previousButtonSegmentation');
    const nextButtonSegmentation = document.getElementById('nextButtonButtonSegmentation');

    let frames = [];
    let currentFrameIndex = 0;
    let currentVideoFile = null;
    let videoDuration = 0;
    let cropCount = 0;
    let segmentationImages = [];
    let currentSegmentationIndex = 0;

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
        const frameSize = document.getElementById('frameSize').value;
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
                frame_size: frameSize,
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
    
            // Espera a imagem carregar para obter suas dimensões
            image.onload = function() {
                const frameWidth = image.width;
                const frameHeight = image.height;

                const originalWidth = image.naturalWidth;
                const originalHeight = image.naturalHeight;

                const originalArea = originalWidth * originalHeight;
                const frameArea = frameWidth * frameHeight;

                // Ajusta o tamanho e posição do cropArea
                const frameSize = document.getElementById('frameSize').value - 3;
                cropArea.style.width = `${frameSize}px`;
                cropArea.style.height = `${frameSize}px`;
                cropArea.style.left = '0px';
                cropArea.style.top = '0px';
            };
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
            cropCount++;
            cropCountElement.textContent = `Cropped images: ${cropCount}`;
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

    function loadSegmentationImages() {
        fetch('/get_processed_images')
            .then(response => response.json())
            .then(data => {
                segmentationImages = data.images;
                if (segmentationImages.length > 0) {
                    loadSegmentationImage(0);
                }
            });
    }

    function loadSegmentationImage(index) {
        if (index >= 0 && index < segmentationImages.length) {
            currentSegmentationIndex = index;
            const img = new Image();
            img.onload = function() {
                segmentationCanvas.width = img.width;
                segmentationCanvas.height = img.height;
                const ctx = segmentationCanvas.getContext('2d');
                ctx.drawImage(img, 0, 0);
            };
            img.src = '/processed/' + segmentationImages[index];
        }
    }

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    segmentationCanvas.addEventListener('mousedown', startDrawing);
    segmentationCanvas.addEventListener('mousemove', draw);
    segmentationCanvas.addEventListener('mouseup', stopDrawing);
    segmentationCanvas.addEventListener('mouseout', stopDrawing);

    function startDrawing(e) {
        isDrawing = true;
        [lastX, lastY] = [e.offsetX, e.offsetY];
    }

    function draw(e) {
        if (!isDrawing) return;
        const ctx = segmentationCanvas.getContext('2d');
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 10;
        ctx.lineCap = 'round';
        ctx.stroke();
        [lastX, lastY] = [e.offsetX, e.offsetY];
    }

    function stopDrawing() {
        isDrawing = false;
    }

    showSegmentationButton.addEventListener('click', () => {

        document.querySelectorAll('.section').forEach(section => {
            section.style.display = 'none';
        });
        
        segmentationSection.style.display = 'block';
        
        loadSegmentationImages();
    });

    previousButtonSegmentation.addEventListener('click', () => {
        if (currentSegmentationIndex > 0) {
            loadSegmentationImage(currentSegmentationIndex - 1);
        }
    });

    nextButtonSegmentation.addEventListener('click', () => {
        if (currentSegmentationIndex < segmentationImages.length - 1) {
            loadSegmentationImage(currentSegmentationIndex + 1);
        }
    });

    applySegmentationButton.addEventListener('click', () => {
        const imageData = segmentationCanvas.toDataURL('image/png');
        fetch('/apply_segmentation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageData,
                filename: segmentationImages[currentSegmentationIndex]
            }),
        })
        .then(response => response.json())
        .then(data => {
            statusMessage.textContent = data.message;
        });
    });
});
