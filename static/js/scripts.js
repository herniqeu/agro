let frames = [];
let currentFrameIndex = 0;
let croppedCount = 0;

document.getElementById('videoInput').addEventListener('change', function (event) {
    const file = event.target.files[0];
    if (file) {
        const formData = new FormData();
        formData.append('file', file);

        fetch('/upload_video', {
            method: 'POST',
            body: formData,
        })
            .then(response => response.text())
            .then(filename => {
                const interval = document.getElementById('interval').value;
                const startTime = document.getElementById('startTime').value;
                const endTime = document.getElementById('endTime').value;

                fetch('/fragment_video', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        filename: filename,
                        interval: interval,
                        start_time: startTime,
                        end_time: endTime
                    }),
                })
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            document.getElementById('statusMessage').innerText = data.error;
                        } else {
                            frames = data.frames;
                            if (frames.length > 0) {
                                loadFrame(0);
                            }
                        }
                    });
            });
    }
});

function loadFrame(index) {
    if (index >= 0 && index < frames.length) {
        currentFrameIndex = index;
        const image = document.getElementById('image');
        image.src = `/frames/${frames[index]}`;
        image.onload = function () {
            document.getElementById('cropArea').style.display = 'block';
        };
        updateCounter();
    }
}

document.getElementById('nextButton').addEventListener('click', function () {
    if (currentFrameIndex < frames.length - 1) {
        loadFrame(currentFrameIndex + 1);
    }
});

document.getElementById('previousButton').addEventListener('click', function () {
    if (currentFrameIndex > 0) {
        loadFrame(currentFrameIndex - 1);
    }
});

document.getElementById('cropButton').addEventListener('click', function () {
    const image = document.getElementById('image');
    const cropRect = cropArea.getBoundingClientRect();
    const imageRect = image.getBoundingClientRect();

    const x = cropRect.left - imageRect.left;
    const y = cropRect.top - imageRect.top;
    const width = cropRect.width;
    const height = cropRect.height;

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
        }),
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById('statusMessage').innerText = data.message;
            croppedCount++;
            updateCounter();
            if (currentFrameIndex < frames.length - 1) {
                loadFrame(currentFrameIndex + 1);  // Move to next frame automatically
            }
        });
});

document.getElementById('skipButton').addEventListener('click', function () {
    loadFrame(currentFrameIndex + 1);
});

function updateCounter() {
    document.getElementById('counterMessage').innerText = `Cropped: ${croppedCount}, Total: ${frames.length}`;
}

const cropArea = document.getElementById('cropArea');
let isDragging = false;
let startX, startY;

cropArea.addEventListener('mousedown', function (event) {
    isDragging = true;
    startX = event.offsetX;
    startY = event.offsetY;
});

cropArea.addEventListener('mousemove', function (event) {
    if (isDragging) {
        const offsetX = event.offsetX - startX;
        const offsetY = event.offsetY - startY;
        cropArea.style.left = `${cropArea.offsetLeft + offsetX}px`;
        cropArea.style.top = `${cropArea.offsetTop + offsetY}px`;
    }
});

cropArea.addEventListener('mouseup', function () {
    isDragging = false;
});