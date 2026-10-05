/**
 * AI Photo Studio — Frontend Engine
 * Minimal, upload-first AI photo editor with coordinate-locked brush cursor,
 * smooth in-place transitions, zero viewport-jump inpainting, and real LaMa integration.
 */

(function () {
  'use strict';

  // API Configuration: Route to local FastAPI backend
  const API_BASE =
    window.location.port === '8000'
      ? window.location.origin
      : 'http://127.0.0.1:8000';
  const REMOVE_OBJECT_ENDPOINT = `${API_BASE}/api/remove-object`;

  // Discrete zoom steps
  const ZOOM_STEPS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0];

  // AI Inference UI Status Stages
  const PROCESSING_STAGES = [
    'Analyzing image',
    'Preparing mask',
    'Reconstructing background',
    'Finishing result'
  ];

  // Core Application State
  const state = {
    // Image references
    initialImage: null,
    currentImage: null,
    resultImage: null,
    resultBlob: null,
    currentFileName: 'image.png',
    imageWidth: 0,
    imageHeight: 0,

    // Tools & Drawing
    currentTool: 'brush', // 'brush' | 'eraser' | 'circle'
    brushSize: 30,        // 5 to 150 px
    isDrawing: false,
    isCircling: false,
    circlePoints: [],
    lastX: 0,
    lastY: 0,
    hasMaskSelection: false,

    // Viewport & Pan/Zoom
    zoom: 1.0,
    fitZoom: 1.0,
    panX: 0,
    panY: 0,
    isPanning: false,
    panStartX: 0,
    panStartY: 0,
    spacePressed: false,

    // Undo / Redo
    undoStack: [],
    redoStack: [],
    maxHistory: 20,

    // Processing & Result State
    isProcessing: false,
    processingStageTimer: null,
    hasResult: false,
    splitSliderPos: 50,
    isDraggingSplit: false,
  };

  // DOM Elements
  const els = {
    // Navigation & Hero
    heroStartBtn: document.getElementById('heroStartBtn'),
    navLoginBtn: document.getElementById('navLoginBtn'),
    navContactBtn: document.getElementById('navContactBtn'),

    // Upload Section
    uploadSection: document.getElementById('uploadSection'),
    dropZone: document.getElementById('dropZone'),
    browseFilesBtn: document.getElementById('browseFilesBtn'),
    fileInput: document.getElementById('fileInput'),

    // Editor Section & Header
    editorSection: document.getElementById('editorSection'),
    imageFilename: document.getElementById('imageFilename'),
    imageDimensions: document.getElementById('imageDimensions'),
    toolNewPhotoBtn: document.getElementById('toolNewPhotoBtn'),

    // Zoom Controls
    zoomOutBtn: document.getElementById('zoomOutBtn'),
    zoomLevelDisplay: document.getElementById('zoomLevelDisplay'),
    zoomInBtn: document.getElementById('zoomInBtn'),
    zoomFitBtn: document.getElementById('zoomFitBtn'),
    zoomOriginalBtn: document.getElementById('zoomOriginalBtn'),

    // Canvas Stage
    canvasViewport: document.getElementById('canvasViewport'),
    canvasContainer: document.getElementById('canvasContainer'),
    imageCanvas: document.getElementById('imageCanvas'),
    resultCanvas: document.getElementById('resultCanvas'),
    maskCanvas: document.getElementById('maskCanvas'),
    circleGuideCanvas: document.getElementById('circleGuideCanvas'),
    splitSlider: document.getElementById('splitSlider'),
    brushCursor: document.getElementById('brushCursor'),

    // Processing Card
    processingStatusCard: document.getElementById('processingStatusCard'),
    processingStageDisplay: document.getElementById('processingStageDisplay'),

    // Right Tools Panel
    toolsPanel: document.getElementById('toolsPanel'),
    toolBrushBtn: document.getElementById('toolBrushBtn'),
    toolCircleBtn: document.getElementById('toolCircleBtn'),
    toolEraserBtn: document.getElementById('toolEraserBtn'),
    toolHint: document.getElementById('toolHint'),
    brushSizeRange: document.getElementById('brushSizeRange'),
    brushSizeDisplay: document.getElementById('brushSizeDisplay'),
    brushPresetPills: document.getElementById('brushPresetPills'),

    // Actions & Buttons
    toolUndoBtn: document.getElementById('toolUndoBtn'),
    toolRedoBtn: document.getElementById('toolRedoBtn'),
    toolClearMaskBtn: document.getElementById('toolClearMaskBtn'),
    removeObjectBtn: document.getElementById('removeObjectBtn'),
    removeBtnSpinner: document.getElementById('removeBtnSpinner'),
    removeBtnIcon: document.getElementById('removeBtnIcon'),
    removeBtnLabel: document.getElementById('removeBtnLabel'),
    actionCaption: document.getElementById('actionCaption'),

    // Result Controls
    resultControls: document.getElementById('resultControls'),
    resultMetrics: document.getElementById('resultMetrics'),
    e2eTimeVal: document.getElementById('e2eTimeVal'),
    e2eDetailedVal: document.getElementById('e2eDetailedVal'),
    serverTimeVal: document.getElementById('serverTimeVal'),
    downloadResultBtn: document.getElementById('downloadResultBtn'),
    downloadBtnLabel: document.getElementById('downloadBtnLabel'),
    editAgainBtn: document.getElementById('editAgainBtn'),
    tryAnotherBtn: document.getElementById('tryAnotherBtn'),

    // Showcase Interactive Slider
    showcaseProductSlider: document.getElementById('showcaseProductSlider'),
    showcaseProductClip: document.getElementById('showcaseProductClip'),
    showcaseProductHandle: document.getElementById('showcaseProductHandle'),

    // Toast Container
    toastContainer: document.getElementById('toastContainer'),
  };

  // Canvas 2D Rendering Contexts
  let imageCtx = null;
  let resultCtx = null;
  let maskCtx = null;
  let circleGuideCtx = null;

  /* ==========================================================================
     1. INITIALIZATION
     ========================================================================== */
  function init() {
    initCanvasContexts();
    attachEventListeners();
    initShowcaseSlider();
    updateUIState();

    // Listen to Auth State changes to safeguard editor view
    if (window.AuthService) {
      window.AuthService.onAuthStateChange((user) => {
        if (!user && els.editorSection && els.editorSection.style.display === 'block') {
          resetEditorToUpload();
        }
      });
    }
  }

  function initCanvasContexts() {
    if (els.imageCanvas) imageCtx = els.imageCanvas.getContext('2d');
    if (els.resultCanvas) resultCtx = els.resultCanvas.getContext('2d');
    if (els.maskCanvas) {
      maskCtx = els.maskCanvas.getContext('2d');
      maskCtx.lineCap = 'round';
      maskCtx.lineJoin = 'round';
    }
    if (els.circleGuideCanvas) {
      circleGuideCtx = els.circleGuideCanvas.getContext('2d');
      circleGuideCtx.lineCap = 'round';
      circleGuideCtx.lineJoin = 'round';
    }
  }

  /* ==========================================================================
     2. EVENT LISTENERS
     ========================================================================== */
  function attachEventListeners() {
    // "Start Creating" CTA directly opens file upload workflow (Auth protected)
    if (els.heroStartBtn) {
      els.heroStartBtn.addEventListener('click', () => {
        if (window.AuthService && !window.AuthService.isAuthenticated()) {
          window.AuthService.openModal(
            'login',
            'Please log in to start creating with AI Photo Studio.',
            () => els.fileInput.click()
          );
          return;
        }
        els.fileInput.click();
      });
    }

    // Top Nav login button
    if (els.navLoginBtn) {
      els.navLoginBtn.addEventListener('click', () => {
        if (window.AuthService) {
          window.AuthService.openModal('login');
        }
      });
    }

    // File Upload (Drag & Drop, Click, File Input)
    if (els.dropZone) {
      els.dropZone.addEventListener('click', (e) => {
        if (e.target !== els.browseFilesBtn) {
          if (window.AuthService && !window.AuthService.isAuthenticated()) {
            window.AuthService.openModal(
              'login',
              'Please log in to upload and edit photos.',
              () => els.fileInput.click()
            );
            return;
          }
          els.fileInput.click();
        }
      });
      els.dropZone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          if (window.AuthService && !window.AuthService.isAuthenticated()) {
            window.AuthService.openModal(
              'login',
              'Please log in to upload and edit photos.',
              () => els.fileInput.click()
            );
            return;
          }
          els.fileInput.click();
        }
      });
    }
    if (els.browseFilesBtn) {
      els.browseFilesBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (window.AuthService && !window.AuthService.isAuthenticated()) {
          window.AuthService.openModal(
            'login',
            'Please log in to upload and edit photos.',
            () => els.fileInput.click()
          );
          return;
        }
        els.fileInput.click();
      });
    }
    if (els.fileInput) {
      els.fileInput.addEventListener('change', handleFileInputChange);
    }

    // Window Drag & Drop
    window.addEventListener('dragover', handleWindowDragOver);
    window.addEventListener('dragleave', handleWindowDragLeave);
    window.addEventListener('drop', handleWindowDrop);

    // Global Clipboard Paste (Ctrl+V)
    window.addEventListener('paste', handleWindowPaste);

    // New Photo / Try Another
    if (els.toolNewPhotoBtn) {
      els.toolNewPhotoBtn.addEventListener('click', () => {
        if (window.AuthService && !window.AuthService.isAuthenticated()) {
          window.AuthService.openModal('login', 'Please log in to upload photos.', () => els.fileInput.click());
          return;
        }
        els.fileInput.click();
      });
    }
    if (els.tryAnotherBtn) {
      els.tryAnotherBtn.addEventListener('click', () => {
        if (window.AuthService && !window.AuthService.isAuthenticated()) {
          window.AuthService.openModal('login', 'Please log in to upload photos.', () => els.fileInput.click());
          return;
        }
        els.fileInput.click();
      });
    }

    // Tool switching (Brush vs Smart Circle vs Eraser)
    if (els.toolBrushBtn) {
      els.toolBrushBtn.addEventListener('click', () => setTool('brush'));
    }
    if (els.toolCircleBtn) {
      els.toolCircleBtn.addEventListener('click', () => setTool('circle'));
    }
    if (els.toolEraserBtn) {
      els.toolEraserBtn.addEventListener('click', () => setTool('eraser'));
    }

    // Brush Size Slider & Presets
    if (els.brushSizeRange) {
      els.brushSizeRange.addEventListener('input', (e) => {
        setBrushSize(parseInt(e.target.value, 10));
      });
    }
    if (els.brushPresetPills) {
      els.brushPresetPills.addEventListener('click', (e) => {
        const btn = e.target.closest('.preset-btn');
        if (btn && btn.dataset.size) {
          setBrushSize(parseInt(btn.dataset.size, 10));
        }
      });
    }

    // Undo / Redo / Clear
    if (els.toolUndoBtn) els.toolUndoBtn.addEventListener('click', undo);
    if (els.toolRedoBtn) els.toolRedoBtn.addEventListener('click', redo);
    if (els.toolClearMaskBtn) els.toolClearMaskBtn.addEventListener('click', clearMaskSelection);

    // Primary AI Removal Action
    if (els.removeObjectBtn) {
      els.removeObjectBtn.addEventListener('click', executeObjectRemoval);
    }

    // Result Actions
    if (els.downloadResultBtn) {
      els.downloadResultBtn.addEventListener('click', handleDownloadResult);
    }
    if (els.editAgainBtn) {
      els.editAgainBtn.addEventListener('click', applyResultAndEditAgain);
    }

    // Zoom Controls
    if (els.zoomInBtn) els.zoomInBtn.addEventListener('click', zoomIn);
    if (els.zoomOutBtn) els.zoomOutBtn.addEventListener('click', zoomOut);
    if (els.zoomFitBtn) els.zoomFitBtn.addEventListener('click', zoomFit);
    if (els.zoomOriginalBtn) els.zoomOriginalBtn.addEventListener('click', zoomOriginal);

    // Canvas Pointer Events (Drawing & Panning)
    if (els.canvasViewport) {
      els.canvasViewport.addEventListener('wheel', handleCanvasWheel, { passive: false });
      els.canvasViewport.addEventListener('pointerdown', handleCanvasPointerDown);
      els.canvasViewport.addEventListener('pointerenter', handleViewportPointerEnter);
      els.canvasViewport.addEventListener('pointerleave', handleViewportPointerLeave);
    }
    window.addEventListener('pointermove', handleWindowPointerMove);
    window.addEventListener('pointerup', handleWindowPointerUp);

    // Split Slider Dragging inside editor
    if (els.splitSlider) {
      const handle = els.splitSlider.querySelector('.slider-divider-handle');
      if (handle) {
        handle.addEventListener('pointerdown', (e) => {
          e.stopPropagation();
          state.isDraggingSplit = true;
          handle.setPointerCapture(e.pointerId);
        });
        handle.addEventListener('pointerup', (e) => {
          state.isDraggingSplit = false;
          try { handle.releasePointerCapture(e.pointerId); } catch (_) {}
        });
      }
    }

    // Keyboard Shortcuts & Window Resize
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    window.addEventListener('resize', handleWindowResize);
  }

  /* ==========================================================================
     3. FILE UPLOAD & TRANSITION
     ========================================================================== */
  function handleFileInputChange(e) {
    const files = e.target.files;
    if (files && files.length > 0) {
      processSelectedFile(files[0]);
    }
    e.target.value = '';
  }

  function handleWindowDragOver(e) {
    e.preventDefault();
    if (els.dropZone) els.dropZone.classList.add('drag-over');
  }

  function handleWindowDragLeave(e) {
    if (e.relatedTarget === null && els.dropZone) {
      els.dropZone.classList.remove('drag-over');
    }
  }

  function handleWindowDrop(e) {
    e.preventDefault();
    if (els.dropZone) els.dropZone.classList.remove('drag-over');
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processSelectedFile(e.dataTransfer.files[0]);
    }
  }

  function handleWindowPaste(e) {
    if (isTypingInInput()) return;
    const items = (e.clipboardData || e.originalEvent.clipboardData)?.items;
    if (!items) return;

    for (let i = 0; i < items.length; i++) {
      if (items[i].type.indexOf('image') !== -1) {
        const file = items[i].getAsFile();
        if (file) {
          processSelectedFile(file);
          showToast('Image pasted from clipboard.', 'info');
          break;
        }
      }
    }
  }

  function processSelectedFile(file) {
    if (!file) return;

    // Check authentication: protected AI editing feature
    if (window.AuthService && !window.AuthService.isAuthenticated()) {
      window.AuthService.openModal(
        'login',
        'Please log in to upload and edit photos.',
        () => processSelectedFile(file)
      );
      return;
    }

    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png|webp)$/i)) {
      showToast('Unsupported format. Please upload JPG, PNG, or WebP.', 'error');
      return;
    }

    if (file.size > 25 * 1024 * 1024) {
      showToast('File size exceeds 25MB limit. Please choose a smaller photo.', 'error');
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        setupEditorWithImage(img, file.name || 'photo.png');
      };
      img.src = event.target.result;
    };
    reader.readAsDataURL(file);
  }

  function resetEditorToUpload() {
    state.initialImage = null;
    state.currentImage = null;
    state.resultImage = null;
    state.resultBlob = null;
    state.hasResult = false;
    state.hasMaskSelection = false;

    const heroSec = document.getElementById('hero');
    if (heroSec) heroSec.style.display = '';
    if (els.uploadSection) els.uploadSection.style.display = '';
    if (els.editorSection) els.editorSection.style.display = 'none';
    if (els.fileInput) els.fileInput.value = '';
    updateUIState();
  }

  function setupEditorWithImage(img, filename) {
    state.initialImage = img;
    state.currentImage = img;
    state.resultImage = null;
    state.resultBlob = null;
    state.currentFileName = filename;
    state.imageWidth = img.naturalWidth || img.width;
    state.imageHeight = img.naturalHeight || img.height;
    state.hasResult = false;

    // Reset stacks & selection
    state.undoStack = [];
    state.redoStack = [];
    state.hasMaskSelection = false;
    state.panX = 0;
    state.panY = 0;

    // Update Header info
    if (els.imageFilename) els.imageFilename.textContent = filename;
    if (els.imageDimensions) {
      els.imageDimensions.textContent = `${state.imageWidth} × ${state.imageHeight} px`;
    }

    // Configure Canvas Dimensions
    setupCanvasLayers();

    // In-Place Smooth Transition: Hide Hero & Upload Section so Editor is immediately visible at the top!
    const heroSec = document.getElementById('hero');
    if (heroSec) heroSec.style.display = 'none';
    if (els.uploadSection) els.uploadSection.style.display = 'none';
    if (els.editorSection) els.editorSection.style.display = 'block';

    // Hide any previous result controls & split slider
    if (els.resultControls) els.resultControls.style.display = 'none';
    if (els.splitSlider) els.splitSlider.style.display = 'none';
    if (els.resultCanvas) els.resultCanvas.classList.remove('active');

    // Default tool
    setTool('brush');

    // Calculate Best Fit Zoom & Center Viewport after DOM reflow
    requestAnimationFrame(() => {
      zoomFit();
      saveUndoState();
      updateUIState();
    });
  }

  function setupCanvasLayers() {
    const w = state.imageWidth;
    const h = state.imageHeight;

    [els.imageCanvas, els.resultCanvas, els.maskCanvas, els.circleGuideCanvas].forEach((canvas) => {
      if (canvas) {
        canvas.width = w;
        canvas.height = h;
      }
    });

    if (imageCtx && state.currentImage) {
      imageCtx.clearRect(0, 0, w, h);
      imageCtx.drawImage(state.currentImage, 0, 0, w, h);
    }

    if (maskCtx) {
      maskCtx.clearRect(0, 0, w, h);
      els.maskCanvas.style.opacity = '1';
    }

    if (circleGuideCtx) {
      circleGuideCtx.clearRect(0, 0, w, h);
    }

    if (els.canvasContainer) {
      els.canvasContainer.style.width = `${w}px`;
      els.canvasContainer.style.height = `${h}px`;
    }
  }

  /* ==========================================================================
     4. TOOLS & BRUSH MANAGEMENT
     ========================================================================== */
  function setTool(toolName) {
    state.currentTool = toolName;

    if (els.toolBrushBtn) {
      els.toolBrushBtn.classList.toggle('active', toolName === 'brush');
      els.toolBrushBtn.setAttribute('aria-selected', toolName === 'brush');
    }
    if (els.toolCircleBtn) {
      els.toolCircleBtn.classList.toggle('active', toolName === 'circle');
      els.toolCircleBtn.setAttribute('aria-selected', toolName === 'circle');
    }
    if (els.toolEraserBtn) {
      els.toolEraserBtn.classList.toggle('active', toolName === 'eraser');
      els.toolEraserBtn.setAttribute('aria-selected', toolName === 'eraser');
    }

    if (els.toolHint) {
      if (toolName === 'circle') {
        els.toolHint.textContent = 'AI Smart Circle: draw a rough loop around any object to auto-detect its boundary.';
      } else if (toolName === 'eraser') {
        els.toolHint.textContent = 'Eraser: brush over selected areas to trim or remove them from the mask.';
      } else {
        els.toolHint.textContent = 'Brush: paint directly over unwanted objects to add them to the mask.';
      }
    }

    if (els.brushCursor) {
      els.brushCursor.classList.remove('eraser-mode', 'circle-mode');
      if (toolName === 'eraser') {
        els.brushCursor.classList.add('eraser-mode');
      } else if (toolName === 'circle') {
        els.brushCursor.classList.add('circle-mode');
      }
    }
  }

  function setBrushSize(size) {
    const clamped = Math.max(5, Math.min(150, size));
    state.brushSize = clamped;

    if (els.brushSizeRange) els.brushSizeRange.value = clamped;
    if (els.brushSizeDisplay) els.brushSizeDisplay.textContent = `${clamped} px`;

    if (els.brushPresetPills) {
      const pills = els.brushPresetPills.querySelectorAll('.preset-btn');
      pills.forEach((p) => {
        p.classList.toggle('active', parseInt(p.dataset.size, 10) === clamped);
      });
    }

    updateBrushCursorSize();
  }

  function updateBrushCursorSize() {
    if (!els.brushCursor) return;
    // Inside canvasContainer, coords match 1:1 image pixels
    els.brushCursor.style.width = `${state.brushSize}px`;
    els.brushCursor.style.height = `${state.brushSize}px`;
  }

  /* ==========================================================================
     5. CANVAS DRAWING & COORDINATES
     ========================================================================== */
  function getCanvasCoords(clientX, clientY) {
    if (!els.maskCanvas) return { x: 0, y: 0 };
    const rect = els.maskCanvas.getBoundingClientRect();
    const scaleX = els.maskCanvas.width / rect.width;
    const scaleY = els.maskCanvas.height / rect.height;
    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
    };
  }

  function handleViewportPointerEnter() {
    if (els.brushCursor && state.currentImage && !state.isPanning) {
      els.brushCursor.style.display = 'block';
    }
  }

  function handleViewportPointerLeave() {
    if (els.brushCursor && !state.isDrawing) {
      els.brushCursor.style.display = 'none';
    }
  }

  function handleCanvasPointerDown(e) {
    if (e.button !== 0) return; // Only primary click

    // Check if middle click or space key pressed -> Pan
    if (state.spacePressed || e.button === 1) {
      state.isPanning = true;
      state.panStartX = e.clientX - state.panX;
      state.panStartY = e.clientY - state.panY;
      els.canvasViewport.classList.add('panning');
      return;
    }

    if (!state.currentImage) return;

    // AI Smart Circle Tool Mode
    if (state.currentTool === 'circle') {
      state.isCircling = true;
      const coords = getCanvasCoords(e.clientX, e.clientY);
      state.circlePoints = [coords];
      clearCircleGuide();
      renderCircleGuide();
      return;
    }

    if (!maskCtx) return;

    state.isDrawing = true;
    saveUndoState();

    const coords = getCanvasCoords(e.clientX, e.clientY);
    state.lastX = coords.x;
    state.lastY = coords.y;

    configureMaskContext();

    // Draw single point
    maskCtx.beginPath();
    maskCtx.arc(coords.x, coords.y, state.brushSize / 2, 0, Math.PI * 2);
    maskCtx.fill();

    state.hasMaskSelection = true;
    updateUIState();
  }

  function handleWindowPointerMove(e) {
    // Handle Split Slider Dragging inside Workspace
    if (state.isDraggingSplit && els.canvasContainer) {
      const rect = els.canvasContainer.getBoundingClientRect();
      const relativeX = e.clientX - rect.left;
      const percent = Math.max(0, Math.min(100, (relativeX / rect.width) * 100));
      updateSplitSliderPosition(percent);
      return;
    }

    // Viewport Panning
    if (state.isPanning) {
      state.panX = e.clientX - state.panStartX;
      state.panY = e.clientY - state.panStartY;
      applyTransform(false);
      return;
    }

    // Update brush cursor position
    if (els.brushCursor && els.maskCanvas) {
      const coords = getCanvasCoords(e.clientX, e.clientY);
      els.brushCursor.style.left = `${coords.x}px`;
      els.brushCursor.style.top = `${coords.y}px`;
    }

    // Smart Circle Guide Drawing
    if (state.isCircling && state.currentImage) {
      const coords = getCanvasCoords(e.clientX, e.clientY);
      const last = state.circlePoints[state.circlePoints.length - 1];
      if (!last || Math.hypot(coords.x - last.x, coords.y - last.y) >= 2.5) {
        state.circlePoints.push(coords);
        renderCircleGuide();
      }
      return;
    }

    // Continuous Brush Stroke Drawing
    if (!state.isDrawing || !maskCtx || !state.currentImage) return;

    const coords = getCanvasCoords(e.clientX, e.clientY);
    configureMaskContext();

    maskCtx.beginPath();
    maskCtx.moveTo(state.lastX, state.lastY);
    maskCtx.lineTo(coords.x, coords.y);
    maskCtx.stroke();

    state.lastX = coords.x;
    state.lastY = coords.y;
    state.hasMaskSelection = true;
  }

  function handleWindowPointerUp() {
    if (state.isPanning) {
      state.isPanning = false;
      els.canvasViewport.classList.remove('panning');
    }

    // Complete Smart Circle selection
    if (state.isCircling) {
      state.isCircling = false;
      const pts = state.circlePoints;
      if (pts && pts.length >= 4) {
        processSmartCircleSelection(pts);
      } else {
        clearCircleGuide();
      }
    }

    if (state.isDrawing) {
      state.isDrawing = false;
      checkMaskSelection();
      updateUIState();
    }
    if (state.isDraggingSplit) {
      state.isDraggingSplit = false;
    }
  }

  /* ==========================================================================
     AI SMART CIRCLE SELECTION & SEGMENTATION
     ========================================================================== */
  function clearCircleGuide() {
    if (circleGuideCtx && state.imageWidth && state.imageHeight) {
      circleGuideCtx.clearRect(0, 0, state.imageWidth, state.imageHeight);
    }
  }

  function renderCircleGuide() {
    if (!circleGuideCtx || !state.circlePoints || state.circlePoints.length < 2) return;
    const pts = state.circlePoints;
    clearCircleGuide();

    circleGuideCtx.save();
    // Modern smartphone AI neon style: glowing cyan outline
    circleGuideCtx.lineCap = 'round';
    circleGuideCtx.lineJoin = 'round';
    circleGuideCtx.lineWidth = 3.5;
    circleGuideCtx.strokeStyle = '#06b6d4';
    circleGuideCtx.shadowColor = 'rgba(6, 182, 212, 0.85)';
    circleGuideCtx.shadowBlur = 10;

    circleGuideCtx.beginPath();
    circleGuideCtx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) {
      circleGuideCtx.lineTo(pts[i].x, pts[i].y);
    }
    circleGuideCtx.stroke();

    // Subtle dashed closing line preview from current pointer to starting origin
    if (pts.length > 5) {
      circleGuideCtx.save();
      circleGuideCtx.lineWidth = 1.8;
      circleGuideCtx.setLineDash([4, 4]);
      circleGuideCtx.strokeStyle = 'rgba(6, 182, 212, 0.6)';
      circleGuideCtx.shadowBlur = 0;
      circleGuideCtx.beginPath();
      circleGuideCtx.moveTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
      circleGuideCtx.lineTo(pts[0].x, pts[0].y);
      circleGuideCtx.stroke();
      circleGuideCtx.restore();
    }

    circleGuideCtx.restore();
  }

  async function processSmartCircleSelection(points) {
    if (!points || points.length < 4 || !state.currentImage || !maskCtx) {
      clearCircleGuide();
      return;
    }

    // Save history before applying new mask
    saveUndoState();

    // Animate temporary glowing fill inside the user's circle
    if (circleGuideCtx) {
      circleGuideCtx.save();
      circleGuideCtx.fillStyle = 'rgba(6, 182, 212, 0.18)';
      circleGuideCtx.beginPath();
      circleGuideCtx.moveTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i++) {
        circleGuideCtx.lineTo(points[i].x, points[i].y);
      }
      circleGuideCtx.closePath();
      circleGuideCtx.fill();
      circleGuideCtx.restore();
    }

    let appliedViaAI = false;

    try {
      // 1. Export base image to PNG blob
      const imageBlob = await new Promise((resolve) => {
        els.imageCanvas.toBlob(resolve, 'image/png');
      });

      // 2. Dispatch to backend AI GrabCut Smart Segmentation endpoint
      const formData = new FormData();
      formData.append('image', imageBlob, 'image.png');
      formData.append('points', JSON.stringify(points));
      formData.append('mode', 'smart_object');

      const response = await fetch(`${API_BASE}/api/smart-circle-segment`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const maskBlob = await response.blob();
        const maskImg = new Image();
        await new Promise((resolve, reject) => {
          maskImg.onload = resolve;
          maskImg.onerror = reject;
          maskImg.src = URL.createObjectURL(maskBlob);
        });

        // Create temporary canvas to inspect and tint the returned segmented mask
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = state.imageWidth;
        tempCanvas.height = state.imageHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(maskImg, 0, 0);

        // Inspect pixel data: strictly tint only true foreground pixels to signature studio red
        // and ensure all unselected pixels remain 100% transparent.
        const imgData = tempCtx.getImageData(0, 0, state.imageWidth, state.imageHeight);
        const d = imgData.data;
        let selectedCount = 0;
        for (let i = 0; i < d.length; i += 4) {
          // A pixel is foreground if luminance is bright OR alpha is active
          const isFg = (d[i] > 100 || d[i + 1] > 100 || d[i + 2] > 100) && (d[i + 3] > 80);
          if (isFg) {
            d[i] = 239;     // R
            d[i + 1] = 68;  // G
            d[i + 2] = 68;  // B
            d[i + 3] = 115; // Signature red mask alpha ~0.45 (115/255)
            selectedCount++;
          } else {
            d[i] = 0;
            d[i + 1] = 0;
            d[i + 2] = 0;
            d[i + 3] = 0;   // Guaranteed 100% transparent background
          }
        }
        tempCtx.putImageData(imgData, 0, 0);

        // Validate that segmentation is non-empty and does not cover the whole photo
        const totalPixels = state.imageWidth * state.imageHeight;
        if (selectedCount > 0 && selectedCount < totalPixels * 0.9) {
          maskCtx.globalCompositeOperation = 'source-over';
          maskCtx.drawImage(tempCanvas, 0, 0);
          appliedViaAI = true;
        }
      }
    } catch (err) {
      console.warn('Backend smart circle segmentation fell back to client polygon fill:', err);
    }

    // Client-side fallback if backend failed or offline
    if (!appliedViaAI) {
      maskCtx.save();
      maskCtx.globalCompositeOperation = 'source-over';
      maskCtx.fillStyle = 'rgba(239, 68, 68, 0.45)';
      maskCtx.strokeStyle = 'rgba(239, 68, 68, 0.45)';
      maskCtx.lineWidth = 4;
      maskCtx.lineJoin = 'round';

      maskCtx.beginPath();
      maskCtx.moveTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i++) {
        maskCtx.lineTo(points[i].x, points[i].y);
      }
      maskCtx.closePath();
      maskCtx.fill();
      maskCtx.stroke();
      maskCtx.restore();
    }

    // Play subtle haptic visual pulse animation on maskCanvas
    if (els.maskCanvas) {
      els.maskCanvas.classList.remove('mask-smart-pulsing');
      void els.maskCanvas.offsetWidth; // trigger reflow
      els.maskCanvas.classList.add('mask-smart-pulsing');
      setTimeout(() => {
        if (els.maskCanvas) els.maskCanvas.classList.remove('mask-smart-pulsing');
      }, 1000);
    }

    state.hasMaskSelection = true;
    clearCircleGuide();
    updateUIState();
    showToast('AI Smart Circle: Object selected! Click "Remove Object" or brush to refine.', 'success');
  }

  function configureMaskContext() {
    if (!maskCtx) return;
    if (state.currentTool === 'eraser') {
      maskCtx.globalCompositeOperation = 'destination-out';
      maskCtx.strokeStyle = 'rgba(0, 0, 0, 1)';
      maskCtx.fillStyle = 'rgba(0, 0, 0, 1)';
    } else {
      maskCtx.globalCompositeOperation = 'source-over';
      maskCtx.strokeStyle = 'rgba(239, 68, 68, 0.45)';
      maskCtx.fillStyle = 'rgba(239, 68, 68, 0.45)';
    }
    maskCtx.lineWidth = state.brushSize;
  }

  function checkMaskSelection() {
    if (!maskCtx || !state.imageWidth || !state.imageHeight) {
      state.hasMaskSelection = false;
      return;
    }
    try {
      const data = maskCtx.getImageData(0, 0, state.imageWidth, state.imageHeight).data;
      let hasPixels = false;
      for (let i = 3; i < data.length; i += 16) {
        if (data[i] > 10) {
          hasPixels = true;
          break;
        }
      }
      state.hasMaskSelection = hasPixels;
    } catch (_) {
      state.hasMaskSelection = false;
    }
  }

  /* ==========================================================================
     6. UNDO / REDO / CLEAR
     ========================================================================== */
  function saveUndoState() {
    if (!maskCtx || !state.imageWidth || !state.imageHeight) return;
    try {
      const snapshot = maskCtx.getImageData(0, 0, state.imageWidth, state.imageHeight);
      state.undoStack.push(snapshot);
      if (state.undoStack.length > state.maxHistory) {
        state.undoStack.shift();
      }
      state.redoStack = [];
      updateUIState();
    } catch (_) {}
  }

  function undo() {
    if (state.undoStack.length === 0 || !maskCtx) return;
    try {
      const current = maskCtx.getImageData(0, 0, state.imageWidth, state.imageHeight);
      state.redoStack.push(current);
      const prev = state.undoStack.pop();
      maskCtx.putImageData(prev, 0, 0);
      checkMaskSelection();
      updateUIState();
    } catch (_) {}
  }

  function redo() {
    if (state.redoStack.length === 0 || !maskCtx) return;
    try {
      const current = maskCtx.getImageData(0, 0, state.imageWidth, state.imageHeight);
      state.undoStack.push(current);
      const next = state.redoStack.pop();
      maskCtx.putImageData(next, 0, 0);
      checkMaskSelection();
      updateUIState();
    } catch (_) {}
  }

  function clearMaskSelection() {
    if (!maskCtx || !state.imageWidth || !state.imageHeight) return;
    saveUndoState();
    maskCtx.clearRect(0, 0, state.imageWidth, state.imageHeight);
    state.hasMaskSelection = false;
    updateUIState();
    showToast('Mask cleared.', 'info');
  }

  /* ==========================================================================
     7. ZOOM & PANNING
     ========================================================================== */
  function zoomIn() {
    const next = ZOOM_STEPS.find((z) => z > state.zoom + 0.05);
    setZoom(next || state.zoom * 1.25, true);
  }

  function zoomOut() {
    const prev = [...ZOOM_STEPS].reverse().find((z) => z < state.zoom - 0.05);
    setZoom(prev || state.zoom * 0.8, true);
  }

  function zoomFit() {
    if (!state.imageWidth || !state.imageHeight || !els.canvasViewport) return;
    const viewW = Math.max(200, els.canvasViewport.clientWidth - 48);
    const viewH = Math.max(200, els.canvasViewport.clientHeight - 48);

    const scaleX = viewW / state.imageWidth;
    const scaleY = viewH / state.imageHeight;
    const fit = Math.min(scaleX, scaleY, 1.0);
    state.fitZoom = fit;
    state.panX = 0;
    state.panY = 0;
    setZoom(fit, true);
  }

  function zoomOriginal() {
    state.panX = 0;
    state.panY = 0;
    setZoom(1.0, true);
  }

  function setZoom(val, smooth = false) {
    const clamped = Math.max(0.1, Math.min(8.0, val));
    state.zoom = clamped;
    if (els.zoomLevelDisplay) {
      els.zoomLevelDisplay.textContent = `${Math.round(clamped * 100)}%`;
    }
    applyTransform(smooth);
    updateBrushCursorSize();
  }

  function applyTransform(smooth = false) {
    if (!els.canvasContainer) return;

    // Constrain pan within reasonable boundaries so image can never be lost
    if (els.canvasViewport && state.imageWidth) {
      const maxPanX = Math.max(200, (state.imageWidth * state.zoom) / 2 + els.canvasViewport.clientWidth / 2);
      const maxPanY = Math.max(200, (state.imageHeight * state.zoom) / 2 + els.canvasViewport.clientHeight / 2);
      state.panX = Math.max(-maxPanX, Math.min(maxPanX, state.panX));
      state.panY = Math.max(-maxPanY, Math.min(maxPanY, state.panY));
    }

    if (smooth) {
      els.canvasContainer.style.transition = 'transform 0.14s cubic-bezier(0.2, 0, 0, 1)';
      setTimeout(() => {
        if (els.canvasContainer) els.canvasContainer.style.transition = '';
      }, 150);
    } else {
      els.canvasContainer.style.transition = '';
    }

    els.canvasContainer.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
  }

  function handleCanvasWheel(e) {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      // Zoom centered on pointer
      const rect = els.canvasViewport.getBoundingClientRect();
      const pointerX = e.clientX - rect.left - rect.width / 2;
      const pointerY = e.clientY - rect.top - rect.height / 2;

      const oldZoom = state.zoom;
      const zoomDelta = e.deltaY < 0 ? 1.15 : 0.87;
      const newZoom = Math.max(0.1, Math.min(8.0, oldZoom * zoomDelta));

      state.panX -= (pointerX - state.panX) * (newZoom / oldZoom - 1);
      state.panY -= (pointerY - state.panY) * (newZoom / oldZoom - 1);
      setZoom(newZoom, false);
    }
    // Note: When no modifier key is held, normal mouse scroll is not intercepted.
    // This allows natural scrolling without flinging the canvas away.
  }

  /* ==========================================================================
     8. AI OBJECT REMOVAL (REAL BACKEND INTEGRATION & NO VIEWPORT JUMP)
     ========================================================================== */
  async function executeObjectRemoval() {
    if (state.isProcessing) return;

    // Check authentication: protected AI editing feature
    if (window.AuthService && !window.AuthService.isAuthenticated()) {
      window.AuthService.openModal(
        'login',
        'Please log in to use AI background reconstruction.',
        () => executeObjectRemoval()
      );
      return;
    }

    if (!state.currentImage) {
      showToast('Please upload an image first.', 'error');
      return;
    }

    if (!state.hasMaskSelection) {
      showToast('Please brush over the object you want to remove.', 'error');
      return;
    }

    // Start latency measurement on client
    const clientStartTime = performance.now();

    try {
      state.isProcessing = true;
      updateUIState();

      // Start non-intrusive in-canvas rotating status stages
      startProcessingStages();

      // Export base image to PNG blob
      const imageBlob = await new Promise((resolve) => {
        els.imageCanvas.toBlob(resolve, 'image/png');
      });

      // Export strict binary mask to PNG blob (white = remove, black = keep)
      const maskBlob = await generateStrictBinaryMaskBlob();

      // Assemble FormData
      const formData = new FormData();
      formData.append('image', imageBlob, 'image.png');
      formData.append('mask', maskBlob, 'mask.png');
      formData.append('prompt', 'seamless background inpainting, photorealistic texture fill');

      // Dispatch request to FastAPI backend
      const response = await fetch(REMOVE_OBJECT_ENDPOINT, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let errMessage = 'Inpainting request failed.';
        try {
          const errData = await response.json();
          errMessage = errData.detail?.message || errData.message || errMessage;
        } catch (_) {
          errMessage = `Server error (${response.status})`;
        }
        throw new Error(errMessage);
      }

      // Read real backend latency header
      const serverProcessTimeMs = response.headers.get('X-Process-Time-Ms');

      // Decode returned result image
      const resultBlob = await response.blob();
      state.resultBlob = resultBlob;

      const resultImg = new Image();
      const resultUrl = URL.createObjectURL(resultBlob);

      await new Promise((resolve, reject) => {
        resultImg.onload = () => resolve();
        resultImg.onerror = () => reject(new Error('Failed to render AI processed image.'));
        resultImg.src = resultUrl;
      });

      state.resultImage = resultImg;
      state.hasResult = true;

      // Stop processing status card
      stopProcessingStages();

      // Render onto Result Canvas
      if (resultCtx) {
        resultCtx.clearRect(0, 0, state.imageWidth, state.imageHeight);
        resultCtx.drawImage(resultImg, 0, 0, state.imageWidth, state.imageHeight);
      }

      if (els.resultCanvas) {
        els.resultCanvas.classList.remove('active');
        void els.resultCanvas.offsetWidth;
        els.resultCanvas.classList.add('active');
      }

      // Enable interactive comparison slider inside SAME workspace
      if (els.splitSlider) els.splitSlider.style.display = 'block';
      updateSplitSliderPosition(50);

      // Smoothly fade out drawn mask
      if (els.maskCanvas) {
        els.maskCanvas.style.transition = 'opacity 0.25s ease';
        els.maskCanvas.style.opacity = '0';
        setTimeout(() => {
          if (maskCtx) maskCtx.clearRect(0, 0, state.imageWidth, state.imageHeight);
          els.maskCanvas.style.opacity = '1';
          els.maskCanvas.style.transition = '';
          state.hasMaskSelection = false;
        }, 260);
      }

      // Compute total client end-to-end time
      const clientEndTime = performance.now();
      const e2eSeconds = ((clientEndTime - clientStartTime) / 1000).toFixed(2);

      // Reveal Result Controls in tools panel with real measurements
      if (els.resultControls) {
        els.resultControls.style.display = 'flex';
      }

      if (els.resultMetrics) {
        if (els.e2eTimeVal) els.e2eTimeVal.textContent = `${e2eSeconds}s`;
        if (els.e2eDetailedVal) els.e2eDetailedVal.textContent = `${e2eSeconds}s`;
        if (els.serverTimeVal) {
          if (serverProcessTimeMs) {
            const serverSec = (parseFloat(serverProcessTimeMs) / 1000).toFixed(2);
            els.serverTimeVal.textContent = `${serverSec}s (${serverProcessTimeMs}ms)`;
          } else {
            els.serverTimeVal.textContent = 'N/A';
          }
        }
      }

      // NOTE: Zero window.scrollTo or element.scrollIntoView called! User stays perfectly centered.
      showToast(`Object removed successfully in ${e2eSeconds}s.`, 'success');
    } catch (err) {
      console.error('Removal Error:', err);
      stopProcessingStages();
      showToast(err.message || 'Unable to process image. Please try again.', 'error');
    } finally {
      state.isProcessing = false;
      updateUIState();
    }
  }

  function startProcessingStages() {
    if (els.maskCanvas) els.maskCanvas.classList.add('mask-processing');
    if (els.processingStatusCard) els.processingStatusCard.style.display = 'flex';

    let currentStageIndex = 0;
    if (els.processingStageDisplay) {
      els.processingStageDisplay.textContent = PROCESSING_STAGES[0];
    }

    if (state.processingStageTimer) clearInterval(state.processingStageTimer);
    state.processingStageTimer = setInterval(() => {
      currentStageIndex = (currentStageIndex + 1) % PROCESSING_STAGES.length;
      if (els.processingStageDisplay) {
        els.processingStageDisplay.textContent = PROCESSING_STAGES[currentStageIndex];
      }
    }, 1800);
  }

  function stopProcessingStages() {
    if (state.processingStageTimer) {
      clearInterval(state.processingStageTimer);
      state.processingStageTimer = null;
    }
    if (els.maskCanvas) els.maskCanvas.classList.remove('mask-processing');
    if (els.processingStatusCard) els.processingStatusCard.style.display = 'none';
  }

  function generateStrictBinaryMaskBlob() {
    return new Promise((resolve) => {
      const w = state.imageWidth;
      const h = state.imageHeight;

      const offscreen = document.createElement('canvas');
      offscreen.width = w;
      offscreen.height = h;
      const offCtx = offscreen.getContext('2d');

      const maskData = maskCtx.getImageData(0, 0, w, h);
      const binaryData = offCtx.createImageData(w, h);

      for (let i = 0; i < maskData.data.length; i += 4) {
        const alpha = maskData.data[i + 3];
        const val = alpha > 25 ? 255 : 0;
        binaryData.data[i] = val;
        binaryData.data[i + 1] = val;
        binaryData.data[i + 2] = val;
        binaryData.data[i + 3] = 255;
      }

      offCtx.putImageData(binaryData, 0, 0);
      offscreen.toBlob((blob) => resolve(blob), 'image/png');
    });
  }

  /* ==========================================================================
     9. BEFORE / AFTER COMPARISON SLIDER (EDITOR)
     ========================================================================== */
  function updateSplitSliderPosition(percent) {
    state.splitSliderPos = percent;
    if (els.splitSlider) {
      els.splitSlider.style.setProperty('--split-percent', `${percent}%`);
      els.splitSlider.style.left = `${percent}%`;
    }
    if (els.resultCanvas) {
      els.resultCanvas.style.setProperty('--split-percent', `${percent}%`);
    }
  }

  /* ==========================================================================
     10. RESULT ACTIONS & DOWNLOAD
     ========================================================================== */
  function handleDownloadResult() {
    const blobToDownload = state.resultBlob;
    if (!blobToDownload && !state.currentImage) {
      showToast('No image available to download.', 'error');
      return;
    }

    const downloadBtn = els.downloadResultBtn;
    const labelSpan = els.downloadBtnLabel;

    if (downloadBtn) downloadBtn.classList.add('downloading');
    if (labelSpan) labelSpan.textContent = 'Preparing...';

    setTimeout(() => {
      try {
        const triggerDownload = (blob) => {
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          const baseName = state.currentFileName.replace(/\.[^/.]+$/, '');
          a.download = `${baseName}-removed.png`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);

          if (labelSpan) labelSpan.textContent = '✓ Downloaded';
          showToast('Image downloaded at original resolution.', 'success');

          setTimeout(() => {
            if (downloadBtn) downloadBtn.classList.remove('downloading');
            if (labelSpan) labelSpan.textContent = 'Download Result';
          }, 1800);
        };

        if (blobToDownload) {
          triggerDownload(blobToDownload);
        } else {
          els.imageCanvas.toBlob((blob) => triggerDownload(blob), 'image/png');
        }
      } catch (err) {
        if (downloadBtn) downloadBtn.classList.remove('downloading');
        if (labelSpan) labelSpan.textContent = 'Download Result';
        showToast('Download failed. Please try again.', 'error');
      }
    }, 100);
  }

  function applyResultAndEditAgain() {
    if (!state.resultImage) return;

    state.currentImage = state.resultImage;
    state.resultImage = null;
    state.resultBlob = null;
    state.hasResult = false;

    if (imageCtx) {
      imageCtx.clearRect(0, 0, state.imageWidth, state.imageHeight);
      imageCtx.drawImage(state.currentImage, 0, 0, state.imageWidth, state.imageHeight);
    }

    if (els.resultCanvas) {
      resultCtx.clearRect(0, 0, state.imageWidth, state.imageHeight);
      els.resultCanvas.classList.remove('active');
    }
    if (els.splitSlider) els.splitSlider.style.display = 'none';
    if (els.resultControls) els.resultControls.style.display = 'none';

    if (maskCtx) maskCtx.clearRect(0, 0, state.imageWidth, state.imageHeight);
    state.undoStack = [];
    state.redoStack = [];
    state.hasMaskSelection = false;
    saveUndoState();

    setTool('brush');
    updateUIState();
    showToast('Ready for next edit. Brush over any object to remove.', 'info');
  }

  /* ==========================================================================
     11. SHOWCASE BEFORE/AFTER COMPARISON SLIDERS (ALL 6 CARDS)
     ========================================================================== */
  function initShowcaseSlider() {
    const containers = document.querySelectorAll('.showcase-slider-container');
    containers.forEach((container) => {
      const handle = container.querySelector('.showcase-slider-handle');
      const imgBefore = container.querySelector('.showcase-img-before');
      if (!handle || !imgBefore) return;

      let isDragging = false;

      function setPosition(xPos) {
        const rect = container.getBoundingClientRect();
        const relativeX = xPos - rect.left;
        const percent = Math.max(0, Math.min(100, (relativeX / rect.width) * 100));
        container.style.setProperty('--slider-split', `${percent}%`);
        handle.style.left = `${percent}%`;
        imgBefore.style.clipPath = `polygon(0 0, ${percent}% 0, ${percent}% 100%, 0 100%)`;
      }

      container.addEventListener('pointerdown', (e) => {
        isDragging = true;
        try { container.setPointerCapture(e.pointerId); } catch (_) {}
        setPosition(e.clientX);
      });

      container.addEventListener('pointermove', (e) => {
        if (!isDragging) return;
        setPosition(e.clientX);
      });

      const stopDrag = (e) => {
        if (isDragging) {
          isDragging = false;
          try { container.releasePointerCapture(e.pointerId); } catch (_) {}
        }
      };

      container.addEventListener('pointerup', stopDrag);
      container.addEventListener('pointercancel', stopDrag);
    });
  }

  /* ==========================================================================
     12. KEYBOARD SHORTCUTS & WINDOW RESIZE
     ========================================================================== */
  function handleKeyDown(e) {
    if (isTypingInInput()) return;

    // Spacebar panning
    if (e.code === 'Space' && !state.spacePressed) {
      state.spacePressed = true;
      els.canvasViewport.classList.add('panning');
    }

    // Undo / Redo
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      if (e.shiftKey) redo();
      else undo();
    } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
      e.preventDefault();
      redo();
    }

    // Tool shortcuts
    else if (e.key.toLowerCase() === 'b' && !e.ctrlKey && !e.metaKey) {
      setTool('brush');
    } else if (e.key.toLowerCase() === 'c' && !e.ctrlKey && !e.metaKey) {
      setTool('circle');
    } else if (e.key.toLowerCase() === 'e' && !e.ctrlKey && !e.metaKey) {
      setTool('eraser');
    }

    // Brush size
    else if (e.key === '[') {
      setBrushSize(state.brushSize - 10);
    } else if (e.key === ']') {
      setBrushSize(state.brushSize + 10);
    }

    // Zoom shortcuts
    else if (e.key === '0' && !e.ctrlKey) {
      zoomFit();
    } else if (e.key === '1' && !e.ctrlKey) {
      zoomOriginal();
    }

    // Enter to run removal
    else if (e.key === 'Enter' && state.hasMaskSelection && !state.isProcessing) {
      executeObjectRemoval();
    }

    // Escape to clear mask
    else if (e.key === 'Escape') {
      if (state.hasMaskSelection) {
        clearMaskSelection();
      }
    }
  }

  function handleKeyUp(e) {
    if (e.code === 'Space') {
      state.spacePressed = false;
      els.canvasViewport.classList.remove('panning');
    }
  }

  function handleWindowResize() {
    if (state.currentImage && state.zoom === state.fitZoom) {
      zoomFit();
    }
  }

  function isTypingInInput() {
    const el = document.activeElement;
    if (!el) return false;
    const tag = el.tagName.toUpperCase();
    return tag === 'INPUT' || tag === 'TEXTAREA' || el.isContentEditable;
  }

  /* ==========================================================================
     13. UI STATE UPDATER
     ========================================================================== */
  function updateUIState() {
    const hasImage = Boolean(state.currentImage);
    const hasSelection = state.hasMaskSelection;
    const isProcessing = state.isProcessing;

    if (els.removeObjectBtn) {
      els.removeObjectBtn.disabled = !hasImage || !hasSelection || isProcessing;
    }

    if (els.removeBtnSpinner) {
      els.removeBtnSpinner.style.display = isProcessing ? 'inline-block' : 'none';
    }
    if (els.removeBtnIcon) {
      els.removeBtnIcon.style.display = isProcessing ? 'none' : 'inline-block';
    }
    if (els.removeBtnLabel) {
      els.removeBtnLabel.textContent = isProcessing
        ? 'Removing Object...'
        : 'Remove Object with AI';
    }

    if (els.actionCaption) {
      if (isProcessing) {
        els.actionCaption.textContent = 'Reconstructing background with AI...';
      } else if (!hasSelection) {
        els.actionCaption.textContent = 'Paint over the object you want to remove.';
      } else {
        els.actionCaption.textContent = 'Ready to remove. Click button or press Enter.';
      }
    }

    if (els.toolUndoBtn) els.toolUndoBtn.disabled = state.undoStack.length === 0;
    if (els.toolRedoBtn) els.toolRedoBtn.disabled = state.redoStack.length === 0;
    if (els.toolClearMaskBtn) els.toolClearMaskBtn.disabled = !hasSelection;
  }

  /* ==========================================================================
     14. TOAST NOTIFICATION HELPER
     ========================================================================== */
  function showToast(message, type = 'info') {
    if (!els.toastContainer) return;
    const toast = document.createElement('div');
    toast.className = `studio-toast ${type}`;
    toast.textContent = message;
    els.toastContainer.appendChild(toast);

    setTimeout(() => toast.classList.add('show'), 10);
    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 250);
    }, 3200);
  }

  // Initialize on DOM Ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
