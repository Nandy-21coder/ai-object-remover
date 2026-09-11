/**
 * AI Object Remover - Frontend Engine
 * Professional canvas editor with sub-pixel coordinate mapping,
 * multi-layer rendering, pan & zoom system, smooth brush/eraser engine,
 * undo/redo history, and real AI inpainting integration.
 */

(function () {
  'use strict';

  // API Endpoints: Ensure requests route to FastAPI backend (port 8000) even when loaded via Live Server
  const API_BASE =
    window.location.port === '8000'
      ? window.location.origin
      : 'http://127.0.0.1:8000';
  const HEALTH_ENDPOINT = `${API_BASE}/api/health`;
  const REMOVE_OBJECT_ENDPOINT = `${API_BASE}/api/remove-object`;

  // Discrete Zoom Levels matching studio standards
  const ZOOM_LEVELS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0];

  // Application State
  const state = {
    initialUploadedImage: null, // Untouched original photo
    currentImage: null,         // Current editing photo (updated after each successful removal)
    originalWidth: 0,
    originalHeight: 0,
    currentTool: 'brush',       // 'brush' | 'eraser' | 'pan'
    previousTool: 'brush',
    brushSize: 30,              // 5px to 200px
    isDrawing: false,
    isPanning: false,
    lastX: 0,
    lastY: 0,
    panX: 0,
    panY: 0,
    panStartX: 0,
    panStartY: 0,
    panOriginX: 0,
    panOriginY: 0,
    zoomLevel: 1.0,
    fitZoom: 1.0,
    history: [],                // Mask snapshots (brush / eraser strokes)
    historyIndex: -1,
    maxHistory: 25,
    imageHistory: [],           // AI result image history stack (Undo Removal)
    imageHistoryIndex: -1,
    isProcessing: false,
    hasMaskSelection: false,
    aiProviderReady: false,
    isSpacePressed: false,
  };

  // DOM Elements
  const els = {
    // Header
    providerStatusBadge: document.getElementById('providerStatusBadge'),

    // Screens
    uploadScreen: document.getElementById('uploadScreen'),
    editorScreen: document.getElementById('editorScreen'),

    // Upload screen
    dropZone: document.getElementById('dropZone'),
    fileInput: document.getElementById('fileInput'),
    browseBtn: document.getElementById('browseBtn'),
    sampleButtons: document.querySelectorAll('.sample-btn'),

    // Editor top bar
    newImageBtn: document.getElementById('newImageBtn'),
    undoRemovalBtn: document.getElementById('undoRemovalBtn'),
    resetToOriginalBtn: document.getElementById('resetToOriginalBtn'),
    imageMetaDimensions: document.getElementById('imageMetaDimensions'),
    undoBtn: document.getElementById('undoBtn'),
    redoBtn: document.getElementById('redoBtn'),
    resetBtn: document.getElementById('resetBtn'),
    zoomOutBtn: document.getElementById('zoomOutBtn'),
    zoomInBtn: document.getElementById('zoomInBtn'),
    zoomResetBtn: document.getElementById('zoomResetBtn'),
    zoomFitBtn: document.getElementById('zoomFitBtn'),
    zoomLevelText: document.getElementById('zoomLevelText'),
    downloadBtn: document.getElementById('downloadBtn'),

    // Tool palette
    toolBrushBtn: document.getElementById('toolBrushBtn'),
    toolEraserBtn: document.getElementById('toolEraserBtn'),
    toolPanBtn: document.getElementById('toolPanBtn'),
    brushSizeSlider: document.getElementById('brushSizeSlider'),
    brushSizeVal: document.getElementById('brushSizeVal'),
    presetPills: document.querySelectorAll('.preset-pill'),
    removeObjectBtn: document.getElementById('removeObjectBtn'),
    maskStatusText: document.getElementById('maskStatusText'),

    // Canvas stage
    viewportContainer: document.getElementById('viewportContainer'),
    canvasWrapper: document.getElementById('canvasWrapper'),
    imageCanvas: document.getElementById('imageCanvas'),
    maskCanvas: document.getElementById('maskCanvas'),
    brushCursor: document.getElementById('brushCursor'),

    // Modals
    processingModal: document.getElementById('processingModal'),
    processingTimer: document.getElementById('processingTimer'),
    progressSteps: document.querySelectorAll('.progress-steps .step-item'),
    alertModal: document.getElementById('alertModal'),
    alertTitle: document.getElementById('alertTitle'),
    alertDescription: document.getElementById('alertDescription'),
    alertInstructionsContainer: document.getElementById('alertInstructionsContainer'),
    alertInstructionsHeader: document.getElementById('alertInstructionsHeader'),
    alertInstructionsCode: document.getElementById('alertInstructionsCode'),
    alertInstructionsNote: document.getElementById('alertInstructionsNote'),
    alertCloseBtn: document.getElementById('alertCloseBtn'),
    alertIconWrapper: document.getElementById('alertIconWrapper'),

    // Toasts
    toastContainer: document.getElementById('toastContainer'),
  };

  // Canvas 2D contexts
  const imgCtx = els.imageCanvas.getContext('2d');
  const maskCtx = els.maskCanvas.getContext('2d');

  /* ==========================================================================
     Initialization & Backend Status Check
     ========================================================================== */
  async function init() {
    setupEventListeners();
    await checkProviderStatus();
  }

  let healthPollingTimer = null;

  async function checkProviderStatus() {
    try {
      const res = await fetch(HEALTH_ENDPOINT);
      if (res.ok) {
        const data = await res.json();
        const isConfigured = !!(data.provider_configured ?? data.configured);
        const providerName = data.provider || data.active_provider || 'AI';
        state.aiProviderReady = isConfigured;
        updateProviderBadge({
          configured: isConfigured,
          active_provider: providerName,
          model: data.model,
        });
        // Keep live sync every 30s once connected
        if (healthPollingTimer) clearTimeout(healthPollingTimer);
        healthPollingTimer = setTimeout(checkProviderStatus, 30000);
      } else {
        updateProviderBadge({ configured: false, active_provider: null });
        if (healthPollingTimer) clearTimeout(healthPollingTimer);
        healthPollingTimer = setTimeout(checkProviderStatus, 3000);
      }
    } catch (err) {
      console.warn('Backend status check error:', err);
      updateProviderBadge({ configured: false, active_provider: null, offline: true });
      // If offline/connecting, retry every 2 seconds automatically
      if (healthPollingTimer) clearTimeout(healthPollingTimer);
      healthPollingTimer = setTimeout(checkProviderStatus, 2000);
    }
  }

  function updateProviderBadge(data) {
    const badge = els.providerStatusBadge;
    badge.className = 'status-badge';
    const textEl = badge.querySelector('.status-text');

    if (data.configured) {
      badge.classList.add('ready');
      textEl.textContent = `AI Engine: ${(data.active_provider || 'AI').toUpperCase()}`;
      badge.title = `Real AI inpainting ready via ${data.active_provider}`;
    } else if (data.offline) {
      badge.classList.add('unconfigured');
      textEl.textContent = 'Backend Offline';
      badge.title = 'FastAPI backend is not reachable.';
    } else {
      badge.classList.add('unconfigured');
      textEl.textContent = 'AI Provider: Not Configured';
      badge.title = 'No API key configured in backend/.env. Real AI required.';
    }
  }

  /* ==========================================================================
     Event Listeners Setup
     ========================================================================== */
  function setupEventListeners() {
    // Upload interactions
    els.browseBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      els.fileInput.click();
    });

    els.dropZone.addEventListener('click', () => els.fileInput.click());

    els.fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleImageFile(e.target.files[0]);
      }
    });

    // Drag & Drop
    ['dragenter', 'dragover'].forEach((eventName) => {
      els.dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        els.dropZone.classList.add('drag-over');
      });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
      els.dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        els.dropZone.classList.remove('drag-over');
      });
    });

    els.dropZone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      if (dt.files && dt.files[0]) {
        handleImageFile(dt.files[0]);
      }
    });

    // Start Creating button in header
    const startCreatingBtn = document.getElementById('startCreatingBtn');
    if (startCreatingBtn) {
      startCreatingBtn.addEventListener('click', () => {
        if (els.fileInput) els.fileInput.click();
      });
    }

    // Mobile nav toggle
    const mobileToggle = document.getElementById('mobileMenuToggle');
    const mobileDrawer = document.getElementById('mobileNavDrawer');
    if (mobileToggle && mobileDrawer) {
      mobileToggle.addEventListener('click', () => {
        mobileDrawer.classList.toggle('open');
      });
    }

    // Sample image buttons
    els.sampleButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        loadSampleImage(btn.dataset.sample);
      });
    });

    // Editor top actions
    els.newImageBtn.addEventListener('click', confirmNewImage);
    if (els.undoRemovalBtn) {
      els.undoRemovalBtn.addEventListener('click', undoRemoval);
    }
    if (els.resetToOriginalBtn) {
      els.resetToOriginalBtn.addEventListener('click', resetToOriginalImage);
    }
    els.undoBtn.addEventListener('click', undo);
    els.redoBtn.addEventListener('click', redo);
    els.resetBtn.addEventListener('click', resetMask);
    els.downloadBtn.addEventListener('click', downloadCurrentResult);

    // Zoom controls
    els.zoomInBtn.addEventListener('click', stepZoomIn);
    els.zoomOutBtn.addEventListener('click', stepZoomOut);
    if (els.zoomResetBtn) {
      els.zoomResetBtn.addEventListener('click', resetZoomTo100);
    }
    els.zoomFitBtn.addEventListener('click', fitImageToViewport);
    els.zoomLevelText.addEventListener('click', resetZoomTo100);

    // Tool switching
    els.toolBrushBtn.addEventListener('click', () => setTool('brush'));
    els.toolEraserBtn.addEventListener('click', () => setTool('eraser'));
    if (els.toolPanBtn) {
      els.toolPanBtn.addEventListener('click', () => setTool('pan'));
    }

    // Brush size slider (5 to 200)
    els.brushSizeSlider.addEventListener('input', (e) => {
      setBrushSize(parseInt(e.target.value, 10));
    });

    // Size presets
    els.presetPills.forEach((pill) => {
      pill.addEventListener('click', () => {
        const sz = parseInt(pill.dataset.size, 10);
        setBrushSize(sz);
      });
    });

    // Action button
    els.removeObjectBtn.addEventListener('click', executeRemoveObject);

    // Modals
    els.alertCloseBtn.addEventListener('click', closeAlertModal);

    // Viewport and Canvas Pointer events
    els.viewportContainer.addEventListener('pointerdown', handleViewportPointerDown);
    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);

    // Viewport wheel for zooming
    els.viewportContainer.addEventListener('wheel', handleWheelZoom, { passive: false });

    // Mouse cursor hover tracking on mask canvas
    els.maskCanvas.addEventListener('mouseenter', () => {
      if (state.currentTool !== 'pan' && !state.isSpacePressed) {
        els.brushCursor.style.display = 'block';
      }
    });
    els.maskCanvas.addEventListener('mouseleave', () => {
      if (!state.isDrawing) {
        els.brushCursor.style.display = 'none';
      }
    });

    // Keyboard shortcuts
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    // Window resize handler
    window.addEventListener('resize', debounce(() => {
      if (state.currentImage && els.editorScreen.classList.contains('active')) {
        updateCanvasDisplayScale();
      }
    }, 120));
  }

  /* ==========================================================================
     Image Loading & Samples
     ========================================================================== */
  function handleImageFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type.toLowerCase())) {
      showToast('Please upload a valid JPG, JPEG, PNG, or WebP image.', 'error');
      return;
    }

    // Max file size: 25MB
    if (file.size > 25 * 1024 * 1024) {
      showToast('Image size exceeds 25MB limit. Please choose a smaller image.', 'error');
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        initializeEditorWithImage(img);
      };
      img.onerror = () => {
        showToast('Failed to load image. The file may be damaged or invalid.', 'error');
      };
      img.src = event.target.result;
    };
    reader.readAsDataURL(file);
  }

  function loadSampleImage(type) {
    const sampleCanvas = document.createElement('canvas');
    sampleCanvas.width = 1200;
    sampleCanvas.height = 800;
    const sCtx = sampleCanvas.getContext('2d');

    if (type === 'beach') {
      // Sky gradient
      const skyGrad = sCtx.createLinearGradient(0, 0, 0, 450);
      skyGrad.addColorStop(0, '#38bdf8');
      skyGrad.addColorStop(0.6, '#bae6fd');
      skyGrad.addColorStop(1, '#fef08a');
      sCtx.fillStyle = skyGrad;
      sCtx.fillRect(0, 0, 1200, 450);

      // Ocean gradient
      const oceanGrad = sCtx.createLinearGradient(0, 450, 0, 560);
      oceanGrad.addColorStop(0, '#0284c7');
      oceanGrad.addColorStop(1, '#0ea5e9');
      sCtx.fillStyle = oceanGrad;
      sCtx.fillRect(0, 450, 1200, 110);

      // Sand
      const sandGrad = sCtx.createLinearGradient(0, 560, 0, 800);
      sandGrad.addColorStop(0, '#fde68a');
      sandGrad.addColorStop(1, '#f59e0b');
      sCtx.fillStyle = sandGrad;
      sCtx.fillRect(0, 560, 1200, 240);

      // Sun
      sCtx.fillStyle = '#fffbeb';
      sCtx.beginPath();
      sCtx.arc(950, 200, 60, 0, Math.PI * 2);
      sCtx.fill();

      // Unwanted object: Wooden signpost right in the middle
      sCtx.fillStyle = '#78350f';
      sCtx.fillRect(580, 460, 20, 240);
      sCtx.fillStyle = '#b45309';
      sCtx.beginPath();
      sCtx.roundRect(500, 430, 180, 60, 8);
      sCtx.fill();
      sCtx.fillStyle = '#ffffff';
      sCtx.font = 'bold 22px sans-serif';
      sCtx.textAlign = 'center';
      sCtx.fillText('PRIVATE AREA', 590, 468);
    } else if (type === 'urban') {
      // City architectural view
      sCtx.fillStyle = '#e2e8f0';
      sCtx.fillRect(0, 0, 1200, 800);

      // Buildings
      sCtx.fillStyle = '#334155';
      sCtx.fillRect(100, 150, 300, 650);
      sCtx.fillStyle = '#475569';
      sCtx.fillRect(450, 80, 350, 720);
      sCtx.fillStyle = '#64748b';
      sCtx.fillRect(850, 220, 280, 580);

      // Windows
      sCtx.fillStyle = '#f8fafc';
      for (let y = 200; y < 750; y += 50) {
        for (let x = 130; x < 370; x += 60) {
          sCtx.fillRect(x, y, 35, 30);
        }
        for (let x = 480; x < 770; x += 60) {
          sCtx.fillRect(x, y, 35, 30);
        }
      }

      // Unwanted object: Construction traffic cone
      sCtx.fillStyle = '#f97316';
      sCtx.beginPath();
      sCtx.moveTo(600, 520);
      sCtx.lineTo(650, 750);
      sCtx.lineTo(550, 750);
      sCtx.closePath();
      sCtx.fill();
      sCtx.fillStyle = '#ffffff';
      sCtx.fillRect(570, 630, 60, 20);
    } else {
      // Studio product mock
      const studioGrad = sCtx.createRadialGradient(600, 400, 50, 600, 400, 600);
      studioGrad.addColorStop(0, '#ffffff');
      studioGrad.addColorStop(1, '#cbd5e1');
      sCtx.fillStyle = studioGrad;
      sCtx.fillRect(0, 0, 1200, 800);

      // Headphone / pedestal
      sCtx.fillStyle = '#0f172a';
      sCtx.beginPath();
      sCtx.ellipse(600, 500, 160, 50, 0, 0, Math.PI * 2);
      sCtx.fill();

      sCtx.fillStyle = '#475569';
      sCtx.beginPath();
      sCtx.arc(600, 380, 90, 0, Math.PI * 2);
      sCtx.fill();

      // Unwanted object: Bright red discount badge
      sCtx.fillStyle = '#ef4444';
      sCtx.beginPath();
      sCtx.roundRect(700, 330, 130, 70, 6);
      sCtx.fill();
      sCtx.fillStyle = '#ffffff';
      sCtx.font = 'bold 20px sans-serif';
      sCtx.textAlign = 'center';
      sCtx.fillText('SALE -50%', 765, 372);
    }

    const img = new Image();
    img.onload = () => initializeEditorWithImage(img);
    img.src = sampleCanvas.toDataURL('image/png');
  }

  /* ==========================================================================
     Editor Initialization & Canvas Setup
     ========================================================================== */
  function initializeEditorWithImage(img) {
    // Preserve untouched copy of the original uploaded image & start AI history stack
    state.initialUploadedImage = img;
    state.currentImage = img;
    state.imageHistory = [img];
    state.imageHistoryIndex = 0;
    state.originalWidth = img.naturalWidth || img.width;
    state.originalHeight = img.naturalHeight || img.height;

    // Set internal resolution of both canvases strictly to original image dimensions
    els.imageCanvas.width = state.originalWidth;
    els.imageCanvas.height = state.originalHeight;

    els.maskCanvas.width = state.originalWidth;
    els.maskCanvas.height = state.originalHeight;

    // Draw base image at full 1:1 resolution
    imgCtx.clearRect(0, 0, state.originalWidth, state.originalHeight);
    imgCtx.drawImage(img, 0, 0, state.originalWidth, state.originalHeight);

    // Clear mask canvas
    maskCtx.clearRect(0, 0, state.originalWidth, state.originalHeight);

    // Reset mask stroke history
    state.history = [];
    state.historyIndex = -1;
    saveHistoryState();

    // Reset pan position
    state.panX = 0;
    state.panY = 0;
    applyPanTransform();

    // Update metadata badge
    els.imageMetaDimensions.textContent = `${state.originalWidth} × ${state.originalHeight} px`;

    // Switch screens
    els.uploadScreen.classList.remove('active');
    els.editorScreen.classList.add('active');

    // Default to brush
    setTool('brush');

    // Fit canvas comfortably into viewport
    setTimeout(() => {
      fitImageToViewport();
      updateButtonsState();
    }, 40);

    showToast('Photo loaded. Brush over the object you want to remove.', 'info');
  }

  function confirmNewImage() {
    if (state.hasMaskSelection) {
      if (!confirm('Open a new image? Current edits and selection will be cleared.')) {
        return;
      }
    }
    els.fileInput.value = '';
    els.editorScreen.classList.remove('active');
    els.uploadScreen.classList.add('active');
    state.hasMaskSelection = false;
    state.initialUploadedImage = null;
    state.currentImage = null;
    updateButtonsState();
  }

  function resetToOriginalImage() {
    if (!state.initialUploadedImage) return;

    if (!confirm('Restore original uploaded image? All subsequent AI removals and selections will be reset.')) {
      return;
    }

    state.currentImage = state.initialUploadedImage;
    state.imageHistory = [state.initialUploadedImage];
    state.imageHistoryIndex = 0;
    state.originalWidth = state.initialUploadedImage.naturalWidth || state.initialUploadedImage.width;
    state.originalHeight = state.initialUploadedImage.naturalHeight || state.initialUploadedImage.height;

    els.imageCanvas.width = state.originalWidth;
    els.imageCanvas.height = state.originalHeight;
    els.maskCanvas.width = state.originalWidth;
    els.maskCanvas.height = state.originalHeight;

    imgCtx.clearRect(0, 0, state.originalWidth, state.originalHeight);
    imgCtx.drawImage(state.initialUploadedImage, 0, 0, state.originalWidth, state.originalHeight);

    maskCtx.clearRect(0, 0, state.originalWidth, state.originalHeight);

    state.history = [];
    state.historyIndex = -1;
    saveHistoryState();

    state.hasMaskSelection = false;
    fitImageToViewport();
    updateButtonsState();

    showToast('Reset to original uploaded photo.', 'success');
  }

  /* ==========================================================================
     Canvas Coordinate Mapping, Display Scaling, Pan & Zoom
     ========================================================================== */
  function fitImageToViewport() {
    if (!state.currentImage) return;

    const vpWidth = els.viewportContainer.clientWidth - 80;
    const vpHeight = els.viewportContainer.clientHeight - 80;

    if (vpWidth <= 0 || vpHeight <= 0) return;

    const aspectImage = state.originalWidth / state.originalHeight;
    const aspectViewport = vpWidth / vpHeight;

    let targetZoom = 1.0;
    if (aspectImage > aspectViewport) {
      targetZoom = vpWidth / state.originalWidth;
    } else {
      targetZoom = vpHeight / state.originalHeight;
    }

    // Keep fit zoom within safe bounds
    targetZoom = Math.min(1.0, Math.max(0.1, targetZoom));

    state.zoomLevel = targetZoom;
    state.fitZoom = targetZoom;
    state.panX = 0;
    state.panY = 0;

    applyPanTransform();
    updateCanvasDisplayScale();
  }

  function resetZoomTo100() {
    state.zoomLevel = 1.0;
    state.panX = 0;
    state.panY = 0;
    applyPanTransform();
    updateCanvasDisplayScale();
  }

  function stepZoomIn() {
    const current = state.zoomLevel;
    // Find next discrete zoom level
    let nextZoom = ZOOM_LEVELS[ZOOM_LEVELS.length - 1];
    for (let i = 0; i < ZOOM_LEVELS.length; i++) {
      if (ZOOM_LEVELS[i] > current + 0.03) {
        nextZoom = ZOOM_LEVELS[i];
        break;
      }
    }
    state.zoomLevel = nextZoom;
    updateCanvasDisplayScale();
  }

  function stepZoomOut() {
    const current = state.zoomLevel;
    // Find previous discrete zoom level
    let prevZoom = ZOOM_LEVELS[0];
    for (let i = ZOOM_LEVELS.length - 1; i >= 0; i--) {
      if (ZOOM_LEVELS[i] < current - 0.03) {
        prevZoom = ZOOM_LEVELS[i];
        break;
      }
    }
    state.zoomLevel = prevZoom;
    updateCanvasDisplayScale();
  }

  function handleWheelZoom(e) {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      if (e.deltaY < 0) {
        stepZoomIn();
      } else {
        stepZoomOut();
      }
    }
  }

  function updateCanvasDisplayScale() {
    if (!state.currentImage) return;

    const dispWidth = Math.round(state.originalWidth * state.zoomLevel);
    const dispHeight = Math.round(state.originalHeight * state.zoomLevel);

    els.canvasWrapper.style.width = `${dispWidth}px`;
    els.canvasWrapper.style.height = `${dispHeight}px`;

    els.imageCanvas.style.width = `${dispWidth}px`;
    els.imageCanvas.style.height = `${dispHeight}px`;

    els.maskCanvas.style.width = `${dispWidth}px`;
    els.maskCanvas.style.height = `${dispHeight}px`;

    els.zoomLevelText.textContent = `${Math.round(state.zoomLevel * 100)}%`;
    updateCursorCircle();
  }

  function applyPanTransform() {
    // Clamp pan to prevent the canvas from disappearing outside view
    const vpW = els.viewportContainer.clientWidth || 800;
    const vpH = els.viewportContainer.clientHeight || 600;
    const dispW = state.originalWidth * state.zoomLevel;
    const dispH = state.originalHeight * state.zoomLevel;

    const maxPanX = Math.max(100, (vpW / 2) + (dispW / 2) - 60);
    const minPanX = -maxPanX;
    const maxPanY = Math.max(100, (vpH / 2) + (dispH / 2) - 60);
    const minPanY = -maxPanY;

    state.panX = Math.max(minPanX, Math.min(maxPanX, state.panX));
    state.panY = Math.max(minPanY, Math.min(maxPanY, state.panY));

    els.canvasWrapper.style.setProperty('--pan-x', `${state.panX}px`);
    els.canvasWrapper.style.setProperty('--pan-y', `${state.panY}px`);
  }

  /**
   * Maps client screen coordinates to exact original internal canvas coordinates.
   * Accurate at any zoom level, pan position, or screen scale.
   */
  function getCanvasCoordinates(e) {
    const rect = els.maskCanvas.getBoundingClientRect();
    const scaleX = state.originalWidth / rect.width;
    const scaleY = state.originalHeight / rect.height;

    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  }

  /* ==========================================================================
     Tool Switching & Brush Sizing
     ========================================================================== */
  function setTool(toolName) {
    state.currentTool = toolName;
    els.toolBrushBtn.classList.toggle('active', toolName === 'brush');
    els.toolEraserBtn.classList.toggle('active', toolName === 'eraser');
    if (els.toolPanBtn) {
      els.toolPanBtn.classList.toggle('active', toolName === 'pan');
    }

    if (toolName === 'pan') {
      els.viewportContainer.classList.add('pan-mode');
      els.brushCursor.style.display = 'none';
    } else {
      els.viewportContainer.classList.remove('pan-mode');
      updateCursorCircle();
    }
  }

  function setBrushSize(size) {
    state.brushSize = Math.max(5, Math.min(200, size));
    els.brushSizeSlider.value = state.brushSize;
    els.brushSizeVal.textContent = `${state.brushSize}px`;

    els.presetPills.forEach((pill) => {
      const pSize = parseInt(pill.dataset.size, 10);
      pill.classList.toggle('active', pSize === state.brushSize);
    });

    updateCursorCircle();
  }

  function updateCursorCircle() {
    if (state.currentTool === 'pan' || state.isSpacePressed) {
      els.brushCursor.style.display = 'none';
      return;
    }

    const rect = els.maskCanvas.getBoundingClientRect();
    if (!rect.width || !state.originalWidth) return;

    // Calculate display diameter corresponding to brush size at current zoom
    const displayDiameter = (state.brushSize * 2) * (rect.width / state.originalWidth);
    els.brushCursor.style.width = `${displayDiameter}px`;
    els.brushCursor.style.height = `${displayDiameter}px`;

    if (state.currentTool === 'eraser') {
      els.brushCursor.style.borderColor = 'rgba(71, 85, 105, 0.95)';
      els.brushCursor.style.backgroundColor = 'rgba(255, 255, 255, 0.3)';
    } else {
      els.brushCursor.style.borderColor = 'rgba(124, 58, 237, 0.95)';
      els.brushCursor.style.backgroundColor = 'rgba(139, 92, 246, 0.3)';
    }
  }

  /* ==========================================================================
     Drawing Engine & Mask Overlay
     ========================================================================== */
  function handleViewportPointerDown(e) {
    if (state.isProcessing) return;

    const isPanAction =
      state.currentTool === 'pan' ||
      state.isSpacePressed ||
      e.button === 1 || // Middle mouse click
      (e.target === els.viewportContainer && e.button === 0);

    if (isPanAction) {
      // Start Panning
      state.isPanning = true;
      state.panStartX = e.clientX;
      state.panStartY = e.clientY;
      state.panOriginX = state.panX;
      state.panOriginY = state.panY;
      els.viewportContainer.classList.add('panning');
      try {
        els.viewportContainer.setPointerCapture(e.pointerId);
      } catch (_) {}
      return;
    }

    // Otherwise, primary button on mask canvas initiates painting
    if (e.button === 0 && (e.target === els.maskCanvas || els.canvasWrapper.contains(e.target))) {
      state.isDrawing = true;
      try {
        els.maskCanvas.setPointerCapture(e.pointerId);
      } catch (_) {}

      const coords = getCanvasCoordinates(e);
      state.lastX = coords.x;
      state.lastY = coords.y;

      drawStroke(coords.x, coords.y, coords.x, coords.y);
    }
  }

  function handlePointerMove(e) {
    if (state.isPanning) {
      const dx = e.clientX - state.panStartX;
      const dy = e.clientY - state.panStartY;
      state.panX = state.panOriginX + dx;
      state.panY = state.panOriginY + dy;
      applyPanTransform();
      return;
    }

    // Update dynamic cursor position relative to canvas wrapper
    const wrapperRect = els.canvasWrapper.getBoundingClientRect();
    const cursorX = e.clientX - wrapperRect.left;
    const cursorY = e.clientY - wrapperRect.top;

    els.brushCursor.style.left = `${cursorX}px`;
    els.brushCursor.style.top = `${cursorY}px`;

    if (!state.isDrawing) return;

    const coords = getCanvasCoordinates(e);
    drawStroke(state.lastX, state.lastY, coords.x, coords.y);
    state.lastX = coords.x;
    state.lastY = coords.y;
  }

  function handlePointerUp(e) {
    if (state.isPanning) {
      state.isPanning = false;
      els.viewportContainer.classList.remove('panning');
      try {
        els.viewportContainer.releasePointerCapture(e.pointerId);
      } catch (_) {}
    }

    if (state.isDrawing) {
      state.isDrawing = false;
      try {
        els.maskCanvas.releasePointerCapture(e.pointerId);
      } catch (_) {}

      saveHistoryState();
      checkIfMaskHasSelection();
    }
  }

  function drawStroke(fromX, fromY, toX, toY) {
    maskCtx.save();
    maskCtx.lineCap = 'round';
    maskCtx.lineJoin = 'round';
    maskCtx.lineWidth = state.brushSize * 2;

    if (state.currentTool === 'brush') {
      // Semi-transparent high-contrast purple overlay
      maskCtx.globalCompositeOperation = 'source-over';
      maskCtx.strokeStyle = 'rgba(139, 92, 246, 0.65)';
      maskCtx.fillStyle = 'rgba(139, 92, 246, 0.65)';
    } else if (state.currentTool === 'eraser') {
      // Eraser removes painted mask pixels cleanly without touching the photo
      maskCtx.globalCompositeOperation = 'destination-out';
      maskCtx.strokeStyle = 'rgba(0, 0, 0, 1)';
      maskCtx.fillStyle = 'rgba(0, 0, 0, 1)';
    }

    maskCtx.beginPath();
    if (fromX === toX && fromY === toY) {
      maskCtx.arc(fromX, fromY, state.brushSize, 0, Math.PI * 2);
      maskCtx.fill();
    } else {
      maskCtx.moveTo(fromX, fromY);
      maskCtx.lineTo(toX, toY);
      maskCtx.stroke();
    }
    maskCtx.restore();
  }

  /* ==========================================================================
     History (Undo / Redo / Reset)
     ========================================================================== */
  function saveHistoryState() {
    // Truncate any redo steps ahead
    if (state.historyIndex < state.history.length - 1) {
      state.history = state.history.slice(0, state.historyIndex + 1);
    }

    // Capture snapshot of current mask canvas
    const imgData = maskCtx.getImageData(0, 0, state.originalWidth, state.originalHeight);
    state.history.push(imgData);

    if (state.history.length > state.maxHistory) {
      state.history.shift();
    } else {
      state.historyIndex++;
    }

    updateButtonsState();
  }

  function undo() {
    if (state.historyIndex > 0) {
      state.historyIndex--;
      const snapshot = state.history[state.historyIndex];
      maskCtx.putImageData(snapshot, 0, 0);
      checkIfMaskHasSelection();
      updateButtonsState();
    }
  }

  function redo() {
    if (state.historyIndex < state.history.length - 1) {
      state.historyIndex++;
      const snapshot = state.history[state.historyIndex];
      maskCtx.putImageData(snapshot, 0, 0);
      checkIfMaskHasSelection();
      updateButtonsState();
    }
  }

  function resetMask() {
    maskCtx.clearRect(0, 0, state.originalWidth, state.originalHeight);
    saveHistoryState();
    checkIfMaskHasSelection();
    showToast('Selection mask cleared', 'info');
  }

  function checkIfMaskHasSelection() {
    const w = state.originalWidth;
    const h = state.originalHeight;
    const imgData = maskCtx.getImageData(0, 0, w, h);
    const data = imgData.data;

    let hasSelection = false;
    // Step scan for any non-zero alpha
    for (let i = 3; i < data.length; i += 16) {
      if (data[i] > 20) {
        hasSelection = true;
        break;
      }
    }

    state.hasMaskSelection = hasSelection;
    updateButtonsState();
  }

  function undoRemoval() {
    if (state.imageHistoryIndex > 0) {
      state.imageHistoryIndex--;
      const prevImg = state.imageHistory[state.imageHistoryIndex];
      state.currentImage = prevImg;

      imgCtx.clearRect(0, 0, state.originalWidth, state.originalHeight);
      imgCtx.drawImage(prevImg, 0, 0, state.originalWidth, state.originalHeight);

      // Clear mask canvas and selection state for next removal
      maskCtx.clearRect(0, 0, state.originalWidth, state.originalHeight);
      state.hasMaskSelection = false;

      // Reset mask history
      state.history = [];
      state.historyIndex = -1;
      saveHistoryState();

      updateButtonsState();
      showToast('Restored photo before last AI removal.', 'info');
    }
  }

  function updateButtonsState() {
    // Mask brush stroke history (History A)
    els.undoBtn.disabled = state.historyIndex <= 0;
    els.redoBtn.disabled = state.historyIndex >= state.history.length - 1;

    // AI removal image history (History B)
    if (els.undoRemovalBtn) {
      els.undoRemovalBtn.disabled = state.imageHistoryIndex <= 0 || state.isProcessing;
    }

    els.removeObjectBtn.disabled = !state.hasMaskSelection || state.isProcessing;

    if (state.hasMaskSelection) {
      els.maskStatusText.textContent = 'Object selected. Ready to remove.';
      els.maskStatusText.style.color = 'var(--accent-primary)';
    } else {
      els.maskStatusText.textContent = 'Brush over an object to select';
      els.maskStatusText.style.color = 'var(--text-muted)';
    }
  }

  /* ==========================================================================
     Binary Mask Generation & AI Inpainting Request
     ========================================================================== */
  /**
   * Generates a strict binary mask PNG Blob matching exact original dimensions:
   * 255 (White) = Area to remove
   * 0 (Black) = Area to preserve
   */
  async function generateBinaryMaskBlob() {
    const w = state.originalWidth;
    const h = state.originalHeight;

    // Strict dimension validation & correction
    if (els.maskCanvas.width !== w || els.maskCanvas.height !== h) {
      console.warn('Correcting mask canvas dimensions to match original image');
      els.maskCanvas.width = w;
      els.maskCanvas.height = h;
    }

    const offCanvas = document.createElement('canvas');
    offCanvas.width = w;
    offCanvas.height = h;
    const offCtx = offCanvas.getContext('2d');

    // 1. Fill background completely black (preserve)
    offCtx.fillStyle = '#000000';
    offCtx.fillRect(0, 0, w, h);

    // 2. Read mask alpha and write strict 255 white onto selected pixels
    const sourceData = maskCtx.getImageData(0, 0, w, h);
    const binaryData = offCtx.createImageData(w, h);

    const src = sourceData.data;
    const dst = binaryData.data;

    for (let i = 0; i < src.length; i += 4) {
      const alpha = src[i + 3];
      if (alpha > 30) {
        // Area to remove -> White
        dst[i] = 255;
        dst[i + 1] = 255;
        dst[i + 2] = 255;
        dst[i + 3] = 255;
      } else {
        // Area to preserve -> Black
        dst[i] = 0;
        dst[i + 1] = 0;
        dst[i + 2] = 0;
        dst[i + 3] = 255;
      }
    }

    offCtx.putImageData(binaryData, 0, 0);

    return new Promise((resolve) => {
      offCanvas.toBlob((blob) => resolve(blob), 'image/png');
    });
  }

  async function getImageBlob() {
    return new Promise((resolve) => {
      els.imageCanvas.toBlob((blob) => resolve(blob), 'image/png');
    });
  }

  async function executeRemoveObject() {
    // Step 1: Validate image
    if (!state.currentImage) {
      showToast('Please upload an image first.', 'error');
      return;
    }

    // Step 2: Validate mask
    if (!state.hasMaskSelection) {
      showToast('Please select the object you want to remove.', 'error');
      return;
    }

    if (state.isProcessing) return;

    // Step 3: Verify dimension alignment
    if (els.imageCanvas.width !== els.maskCanvas.width || els.imageCanvas.height !== els.maskCanvas.height) {
      showToast('Internal mask dimension mismatch. Correcting...', 'info');
      els.maskCanvas.width = state.originalWidth;
      els.maskCanvas.height = state.originalHeight;
    }

    state.isProcessing = true;
    updateButtonsState();

    // Step 1: Preparing selection
    showProcessingModal();
    setModalStep(1);

    let timeoutId = null;

    try {
      await sleep(200);

      // Step 2: Optimizing mask
      setModalStep(2);

      const [imageBlob, maskBlob] = await Promise.all([
        getImageBlob(),
        generateBinaryMaskBlob(),
      ]);

      const formData = new FormData();
      formData.append('image', imageBlob, 'image.png');
      formData.append('mask', maskBlob, 'mask.png');
      formData.append('prompt', 'seamless photo restoration, clean background infill');

      const controller = new AbortController();
      // Increase timeout to 300s (5 minutes) for high-resolution CPU neural inpainting
      timeoutId = setTimeout(() => controller.abort(), 300000);

      // Step 3: AI reconstructing background
      setModalStep(3);

      const response = await fetch(REMOVE_OBJECT_ENDPOINT, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });

      if (timeoutId) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }

      if (response.status === 503) {
        hideProcessingModal();
        let errJson = {};
        try {
          errJson = await response.json();
        } catch (_) {}
        showAlertModal({
          title: 'AI Provider Not Configured',
          description:
            errJson.detail?.message ||
            'Real generative AI inpainting is required to remove objects.',
          instructions:
            errJson.detail?.instructions ||
            'Add your AI_API_KEY in backend/.env to connect a real inpainting provider.',
          headerTitle: 'backend/.env',
          isError: false,
        });
        return;
      }

      if (!response.ok) {
        // Section 12 Quality Fallback:
        // Do NOT use blur, gradient, or cloning fallback.
        // Keep current image and mask.
        hideProcessingModal();
        let errMsg = 'AI removal failed. Please try again or adjust your selection.';
        let errTitle = 'Inpainting Failed';
        let instructions = 'Check your selection or API credentials in backend/.env.';
        let headerTitle = 'Troubleshooting';

        try {
          const errData = await response.json();
          if (errData.detail) {
            if (typeof errData.detail === 'object') {
              errMsg = errData.detail.message || errMsg;
              if (errData.detail.instructions) instructions = errData.detail.instructions;
            } else {
              errMsg = errData.detail;
            }
          }
          if (response.status === 401) {
            errTitle = 'Invalid API Key';
            instructions = 'Check your AI_API_KEY in backend/.env.';
            headerTitle = 'backend/.env';
          } else if (response.status === 422) {
            errTitle = 'Invalid Selection';
            instructions = 'Please brush over the object you wish to remove.';
            headerTitle = 'Selection Tip';
          } else if (response.status === 429) {
            errTitle = 'Rate Limit Exceeded';
            instructions = 'Provider quota or rate limit exceeded. Please wait a moment.';
            headerTitle = 'Rate Limit';
          } else if (response.status === 504) {
            errTitle = 'Request Timed Out';
            instructions = 'The local AI inpainting engine took longer than the server timeout limit.\n\nTips:\n- Select a tighter brush area around the object to process faster.\n- Close high-CPU background tasks and retry.';
            headerTitle = 'Performance Tips';
          }
        } catch (_) {
          errMsg = `Server error (${response.status}): ${response.statusText}`;
        }

        showToast('AI removal failed. Please try again or adjust your selection.', 'error');
        showAlertModal({
          title: errTitle,
          description: errMsg,
          instructions: instructions,
          headerTitle: headerTitle,
          isError: true,
        });
        return;
      }

      // Step 4: Finalizing result
      setModalStep(4);
      await sleep(150);

      const resultBlob = await response.blob();
      const resultUrl = URL.createObjectURL(resultBlob);

      const resultImage = new Image();
      resultImage.onload = () => {
        // Redraw image canvas with result
        imgCtx.clearRect(0, 0, state.originalWidth, state.originalHeight);
        imgCtx.drawImage(resultImage, 0, 0, state.originalWidth, state.originalHeight);

        // Update current editing image and record in AI Removal History (History B)
        state.currentImage = resultImage;
        state.imageHistory.push(resultImage);
        state.imageHistoryIndex = state.imageHistory.length - 1;

        // Clear mask canvas and selection state (ready for next object selection)
        maskCtx.clearRect(0, 0, state.originalWidth, state.originalHeight);
        state.hasMaskSelection = false;

        // Reset mask stroke history (History A)
        state.history = [];
        state.historyIndex = -1;
        saveHistoryState();

        hideProcessingModal();
        URL.revokeObjectURL(resultUrl);
        showToast('Object removed successfully with AI!', 'success');
      };
      resultImage.src = resultUrl;
    } catch (err) {
      hideProcessingModal();
      showToast('AI removal failed. Please try again or adjust your selection.', 'error');
      if (err.name === 'AbortError') {
        showAlertModal({
          title: 'Request Timed Out',
          description: 'The AI inpainting process took longer than expected. Your painted selection has been preserved.',
          instructions: 'The local AI inpainting engine is processing on CPU or system load was high.\n\nTips to resolve:\n1. Click "Remove Object" again to retry.\n2. Selecting a slightly smaller or more targeted area processes faster.\n3. Ensure high CPU background apps are closed for best speed.',
          headerTitle: 'Performance & Troubleshooting Tips',
          note: 'Your drawn mask is saved on the canvas. Click Remove Object to try again.',
          isError: true,
        });
      } else {
        showAlertModal({
          title: 'Connection Error',
          description: `Could not complete AI request: ${err.message}`,
          instructions: 'Ensure the FastAPI backend server is running.\n\nTo start or restart the server:\nRun start_server.bat in the ai-object-remover directory.',
          headerTitle: 'Server Troubleshooting',
          note: 'Your drawn mask has been preserved. Reconnect the server and click Remove Object.',
          isError: true,
        });
      }
    } finally {
      if (timeoutId) clearTimeout(timeoutId);
      state.isProcessing = false;
      updateButtonsState();
    }
  }

  /* ==========================================================================
     Download Flow
     ========================================================================== */
  function downloadCurrentResult() {
    if (!state.currentImage) return;

    // Export the image canvas at full 1:1 resolution
    els.imageCanvas.toBlob((blob) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `ai-object-remover-${Date.now()}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      showToast('Image downloaded at full resolution.', 'success');
    }, 'image/png');
  }

  /* ==========================================================================
     Modal & Toast Helpers
     ========================================================================== */
  let processingStartTime = 0;
  let processingTimerInterval = null;

  function showProcessingModal() {
    processingStartTime = Date.now();
    if (els.processingTimer) {
      els.processingTimer.textContent = 'Processing with AI... Elapsed: 0s';
    }
    if (processingTimerInterval) clearInterval(processingTimerInterval);
    processingTimerInterval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - processingStartTime) / 1000);
      if (els.processingTimer) {
        els.processingTimer.textContent = `Processing with AI... Elapsed: ${elapsed}s`;
      }
    }, 1000);
    els.processingModal.classList.add('active');
  }

  function hideProcessingModal() {
    if (processingTimerInterval) {
      clearInterval(processingTimerInterval);
      processingTimerInterval = null;
    }
    els.processingModal.classList.remove('active');
  }

  function setModalStep(stepNum) {
    els.progressSteps.forEach((step) => {
      const n = parseInt(step.dataset.step, 10);
      step.classList.remove('active', 'completed');
      if (n < stepNum) {
        step.classList.add('completed');
        step.querySelector('.step-bullet').innerHTML = '&#10003;';
      } else if (n === stepNum) {
        step.classList.add('active');
        step.querySelector('.step-bullet').textContent = n;
      } else {
        step.querySelector('.step-bullet').textContent = n;
      }
    });
  }

  function showAlertModal({ title, description, instructions, isError, headerTitle, note }) {
    els.alertTitle.textContent = title;
    els.alertDescription.textContent = description;

    if (instructions) {
      els.alertInstructionsContainer.style.display = 'block';
      els.alertInstructionsCode.textContent = instructions;
      if (els.alertInstructionsHeader) {
        els.alertInstructionsHeader.textContent = headerTitle || (isError ? 'Troubleshooting Tips' : 'backend/.env');
      }
      if (els.alertInstructionsNote) {
        if (note) {
          els.alertInstructionsNote.textContent = note;
          els.alertInstructionsNote.style.display = 'block';
        } else if (headerTitle && headerTitle !== 'backend/.env') {
          els.alertInstructionsNote.textContent = 'Your drawn mask has been safely preserved on the canvas. Click Remove Object to try again.';
          els.alertInstructionsNote.style.display = 'block';
        } else {
          els.alertInstructionsNote.textContent = 'Your drawn mask has been safely preserved on the canvas. Once configured, click Remove Object to perform real inpainting.';
          els.alertInstructionsNote.style.display = 'block';
        }
      }
    } else {
      els.alertInstructionsContainer.style.display = 'none';
    }

    if (isError) {
      els.alertIconWrapper.className = 'alert-icon-wrapper error';
    } else {
      els.alertIconWrapper.className = 'alert-icon-wrapper info';
    }

    els.alertModal.classList.add('active');
  }

  function closeAlertModal() {
    els.alertModal.classList.remove('active');
  }

  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    let iconSvg = '';
    if (type === 'error') {
      iconSvg = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`;
    } else if (type === 'success') {
      iconSvg = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>`;
    } else {
      iconSvg = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`;
    }

    toast.innerHTML = `${iconSvg}<span>${escapeHtml(message)}</span>`;
    els.toastContainer.appendChild(toast);

    setTimeout(() => toast.classList.add('show'), 10);
    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 250);
    }, 4000);
  }

  /* ==========================================================================
     Keyboard Shortcuts
     ========================================================================== */
  function isTypingInInput() {
    const activeEl = document.activeElement;
    if (!activeEl) return false;
    const tag = activeEl.tagName.toUpperCase();
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || activeEl.isContentEditable;
  }

  function handleKeyDown(e) {
    if (isTypingInInput()) return;
    if (!els.editorScreen.classList.contains('active')) return;

    // Spacebar temporary Pan mode
    if (e.code === 'Space' && !state.isSpacePressed) {
      e.preventDefault();
      state.isSpacePressed = true;
      state.previousTool = state.currentTool;
      setTool('pan');
      return;
    }

    // Ctrl + Z -> Undo, Ctrl + Y / Ctrl + Shift + Z -> Redo
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      if (e.shiftKey) {
        redo();
      } else {
        undo();
      }
    } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
      e.preventDefault();
      redo();
    }
    // Ctrl + + / Ctrl + = -> Zoom In
    else if ((e.ctrlKey || e.metaKey) && (e.key === '+' || e.key === '=')) {
      e.preventDefault();
      stepZoomIn();
    }
    // Ctrl + - -> Zoom Out
    else if ((e.ctrlKey || e.metaKey) && e.key === '-') {
      e.preventDefault();
      stepZoomOut();
    }
    // R or B -> Brush
    else if (e.key.toLowerCase() === 'r' || e.key.toLowerCase() === 'b') {
      if (!e.ctrlKey && !e.metaKey) {
        setTool('brush');
      }
    }
    // E -> Eraser
    else if (e.key.toLowerCase() === 'e') {
      if (!e.ctrlKey && !e.metaKey) {
        setTool('eraser');
      }
    }
    // H -> Pan/Hand
    else if (e.key.toLowerCase() === 'h') {
      if (!e.ctrlKey && !e.metaKey) {
        setTool('pan');
      }
    }
    // Bracket keys to adjust brush size
    else if (e.key === '[') {
      setBrushSize(state.brushSize - 10);
    } else if (e.key === ']') {
      setBrushSize(state.brushSize + 10);
    }
  }

  function handleKeyUp(e) {
    if (e.code === 'Space' && state.isSpacePressed) {
      state.isSpacePressed = false;
      setTool(state.previousTool || 'brush');
    }
  }

  /* ==========================================================================
     Utility Functions
     ========================================================================== */
  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function debounce(fn, wait) {
    let timeout;
    return function (...args) {
      clearTimeout(timeout);
      timeout = setTimeout(() => fn.apply(this, args), wait);
    };
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  // Run on DOM loaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
