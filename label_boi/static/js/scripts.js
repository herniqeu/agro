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

    const showSegmentationButton = document.getElementById('showSegmentationButton');

    function showSegmentationSection() {
        console.log("Botão Segmentation clicado");
        document.querySelectorAll('.section').forEach(section => {
            section.style.display = 'none';
        });
        segmentationSection.style.display = 'block';
        loadSegmentationImages();
    }

    function loadSegmentationImages() {
        console.log("Carregando imagens de segmentação");
        fetch('/get_processed_images')
            .then(response => response.json())
            .then(data => {
                segmentationImages = data.images;
                console.log("Imagens carregadas:", segmentationImages);
                if (segmentationImages.length > 0) {
                    loadSegmentationImage(0);
                } else {
                    console.log("Nenhuma imagem processada encontrada.");
                    statusMessage.textContent = "Nenhuma imagem processada encontrada.";
                }
            })
            .catch(error => {
                console.error("Erro ao carregar imagens:", error);
                statusMessage.textContent = "Erro ao carregar imagens.";
            });
    }

    showSegmentationButton.addEventListener('click', showSegmentationSection);

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
    
        const loadingInterval = animateLoadingText('statusMessage', 'Uploading video');
    
        fetch('/upload_video', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            clearInterval(loadingInterval);
            if (data.error) {
                statusMessage.textContent = data.error;
            } else {
                currentVideoFile = data.filename;
                videoDuration = data.duration;
                initializeTimeSlider();
                uploadSection.style.display = 'none';
                configSection.style.display = 'block';
                statusMessage.textContent = '';
            }
        })
        .catch(error => {
            clearInterval(loadingInterval);
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
        const framesPerSecond = document.getElementById('framesPerSecond').value;
        const frameSize = document.getElementById('frameSize').value;
        const startTime = $("#timeSlider").slider("values", 0);
        const endTime = $("#timeSlider").slider("values", 1);
    
        const loadingInterval = animateLoadingText('statusMessage', 'Loading process');
    
        fetch('/fragment_video', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                filename: currentVideoFile,
                frames_per_second: framesPerSecond,
                frame_size: frameSize,
                start_time: startTime,
                end_time: endTime
            }),
        })
        .then(response => response.json())
        .then(data => {
            clearInterval(loadingInterval);
            if (data.error) {
                statusMessage.textContent = data.error;
            } else {
                frames = data.frames;
                configSection.style.display = 'none';
                cropSection.style.display = 'block';
                loadFrame(0);
                statusMessage.textContent = '';
            }
        })
        .catch(error => {
            clearInterval(loadingInterval);
            statusMessage.textContent = 'Error processing video: ' + error;
        });
    });

    function animateLoadingText(elementId, baseText) {
        let dots = 0;
        return setInterval(() => {
            const element = document.getElementById(elementId);
            dots = (dots + 1) % 4;
            element.textContent = baseText + '.'.repeat(dots);
        }, 500);
    }

    function loadFrame(index) {
        if (index >= 0 && index < frames.length) {
            currentFrameIndex = index;
            image.src = '/frames/' + frames[index];
    
            image.onload = function() {
                const containerWidth = 600;
                const containerHeight = 400;
                
                const originalWidth = image.naturalWidth;
                const originalHeight = image.naturalHeight;
                
                const scale = Math.min(containerWidth / originalWidth, containerHeight / originalHeight);
                
                const scaledWidth = originalWidth * scale;
                const scaledHeight = originalHeight * scale;
                
                image.style.width = `${scaledWidth}px`;
                image.style.height = `${scaledHeight}px`;
    
                const frameSize = document.getElementById('frameSize').value;
                const scaledFrameSize = frameSize * scale;
                
                cropArea.style.width = `${scaledFrameSize}px`;
                cropArea.style.height = `${scaledFrameSize}px`;
                cropArea.style.left = '0px';
                cropArea.style.top = '0px';
    
                // Store original dimensions and scale for use in cropping
                image.dataset.originalWidth = originalWidth;
                image.dataset.originalHeight = originalHeight;
                image.dataset.scale = scale;
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

    document.getElementById('brushSize').addEventListener('input', function() {
        document.getElementById('brushSizeValue').textContent = this.value;
    });

    cropButton.addEventListener('click', () => {
        const rect = cropArea.getBoundingClientRect();
        const imageRect = image.getBoundingClientRect();
    
        const scale = parseFloat(image.dataset.scale);
        const originalWidth = parseInt(image.dataset.originalWidth);
        const originalHeight = parseInt(image.dataset.originalHeight);
    
        const x = (rect.left - imageRect.left) / scale;
        const y = (rect.top - imageRect.top) / scale;
        const width = rect.width / scale;
        const height = rect.height / scale;
    
        fetch('/crop', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                filename: frames[currentFrameIndex],
                x: x,
                y: y,
                width: width,
                height: height,
                originalWidth: originalWidth,
                originalHeight: originalHeight
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

    function loadSegmentationImage(index) {
        if (index >= 0 && index < segmentationImages.length) {
            currentSegmentationIndex = index;
            const img = new Image();
            img.onload = function() {
                const containerWidth = 600;
                const containerHeight = 400;
                
                const originalWidth = img.naturalWidth;
                const originalHeight = img.naturalHeight;
                
                const scale = Math.min(containerWidth / originalWidth, containerHeight / originalHeight);
                
                const scaledWidth = originalWidth * scale;
                const scaledHeight = originalHeight * scale;
                
                segmentationCanvas.width = scaledWidth;
                segmentationCanvas.height = scaledHeight;
                const ctx = segmentationCanvas.getContext('2d');
                ctx.drawImage(img, 0, 0, scaledWidth, scaledHeight);
    
                // Store original dimensions and scale for use in segmentation
                segmentationCanvas.dataset.originalWidth = originalWidth;
                segmentationCanvas.dataset.originalHeight = originalHeight;
                segmentationCanvas.dataset.scale = scale;
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
        ctx.lineWidth = document.getElementById('brushSize').value;
        ctx.lineCap = 'round';
        ctx.stroke();
        [lastX, lastY] = [e.offsetX, e.offsetY];
    }

    function stopDrawing() {
        isDrawing = false;
    }

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
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const originalWidth = parseInt(segmentationCanvas.dataset.originalWidth);
        const originalHeight = parseInt(segmentationCanvas.dataset.originalHeight);
        
        canvas.width = originalWidth;
        canvas.height = originalHeight;
        
        ctx.drawImage(segmentationCanvas, 0, 0, originalWidth, originalHeight);
        
        const imageData = canvas.toDataURL('image/png');
        fetch('/apply_segmentation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageData,
                filename: segmentationImages[currentSegmentationIndex],
                originalWidth: originalWidth,
                originalHeight: originalHeight
            }),
        })
        .then(response => response.json())
        .then(data => {
            statusMessage.textContent = data.message;
        });
    });
});
