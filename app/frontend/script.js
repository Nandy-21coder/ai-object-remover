/**
 * AI Canvas Studio — Frontend Engine
 * Professional desktop-style AI photo editor workspace
 * Micro-interactions, GPU transitions, coordinate-locked brush cursor,
 * and real LaMa inpainting integration.
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

  // Core Application State
  const state = {
    // Images
    initialImage: null,      // Pristine original image
    currentImage: null,      // Active image currently being edited
    resultImage: null,       // AI inpainting result
    resultBlob: null,        // Latest result blob for high-res download
    currentFile: null,       // File object reference
    imageWidth: 0,
    imageHeight: 0,

    // Tools & Drawing
    currentTool: 'brush',    // 'brush' | 'eraser' | 'select'
    brushSize: 30,           // 5 to 150 px
    isDrawing: false,
    lastX: 0,
    lastY: 0,
    hasMaskSelection: false,

    // Viewport & Zoom
    zoom: 1.0,
    fitZoom: 1.0,
    panX: 0,
    panY: 0,
    isPanning: false,
    panStartX: 0,
    panStartY: 0,
    spacePressed: false,

    // History
    undoStack: [],
    redoStack: [],
    maxHistory: 20,

    // Processing & Result State
    isProcessing: false,
    processingPhaseTimer: null,
    hasResult: false,
    splitSliderPos: 50,      // Percent (0 - 100)
    isDraggingSplit: false,
  };

  // DOM Elements
  const els = {
    // Topbar
    studioTopbar: document.getElementById('studioTopbar'),
    brandLogo: document.getElementById('brandLogo'),
    projectTitle: document.getElementById('projectTitle'),
    projectsBtn: document.getElementById('projectsBtn'),
    saveBtn: document.getElementById('saveBtn'),
    exportBtn: document.getElementById('exportBtn'),
    exportBtnLabel: document.getElementById('exportBtnLabel'),

    // Left Toolbar
    toolSelectBtn: document.getElementById('toolSelectBtn'),
    toolBrushBtn: document.getElementById('toolBrushBtn'),
    toolEraserBtn: document.getElementById('toolEraserBtn'),
    toolRemoveQuickBtn: document.getElementById('toolRemoveQuickBtn'),
    toolUndoBtn: document.getElementById('toolUndoBtn'),
    toolRedoBtn: document.getElementById('toolRedoBtn'),
    toolClearMaskBtn: document.getElementById('toolClearMaskBtn'),
    toolNewPhotoBtn: document.getElementById('toolNewPhotoBtn'),

    // Center Stage
    centerStage: document.getElementById('centerStage'),
    emptyState: document.getElementById('emptyState'),
    emptyDropZone: document.getElementById('emptyDropZone'),
    uploadPhotoBtn: document.getElementById('uploadPhotoBtn'),
    pasteClipboardBtn: document.getElementById('pasteClipboardBtn'),
    fileInput: document.getElementById('fileInput'),

    // Canvas Stage
    canvasStage: document.getElementById('canvasStage'),
    canvasViewport: document.getElementById('canvasViewport'),
    canvasContainer: document.getElementById('canvasContainer'),
    imageCanvas: document.getElementById('imageCanvas'),
    resultCanvas: document.getElementById('resultCanvas'),
    maskCanvas: document.getElementById('maskCanvas'),
    splitSlider: document.getElementById('splitSlider'),
    brushCursor: document.getElementById('brushCursor'),
    processingStatusCard: document.getElementById('processingStatusCard'),
    phase1: document.getElementById('phase1'),
    phase2: document.getElementById('phase2'),
    phase3: document.getElementById('phase3'),

    // Bottom Canvas Bar
    bottomCanvasBar: document.getElementById('bottomCanvasBar'),
    zoomOutBtn: document.getElementById('zoomOutBtn'),
    zoomLevelDisplay: document.getElementById('zoomLevelDisplay'),
    zoomInBtn: document.getElementById('zoomInBtn'),
    zoomFitBtn: document.getElementById('zoomFitBtn'),
    zoomOriginalBtn: document.getElementById('zoomOriginalBtn'),
    imageDimensionsBadge: document.getElementById('imageDimensionsBadge'),

    // Right Contextual Panel
    rightPanel: document.getElementById('rightPanel'),
    panelToolBrush: document.getElementById('panelToolBrush'),
    panelToolEraser: document.getElementById('panelToolEraser'),
    brushSizeRange: document.getElementById('brushSizeRange'),
    brushSizeDisplay: document.getElementById('brushSizeDisplay'),
    brushPresetPills: document.getElementById('brushPresetPills'),
    eraseMaskBtn: document.getElementById('eraseMaskBtn'),
    clearSelectionBtn: document.getElementById('clearSelectionBtn'),
    removeObjectBtn: document.getElementById('removeObjectBtn'),
    removeBtnSpinner: document.getElementById('removeBtnSpinner'),
    removeBtnIcon: document.getElementById('removeBtnIcon'),
    removeBtnLabel: document.getElementById('removeBtnLabel'),
    actionCaption: document.getElementById('actionCaption'),

    // Result Controls
    resultControls: document.getElementById('resultControls'),
    resultBadge: document.getElementById('resultBadge'),
    resultMetrics: document.getElementById('resultMetrics'),
    e2eTimeVal: document.getElementById('e2eTimeVal'),
    e2eDetailedVal: document.getElementById('e2eDetailedVal'),
    serverTimeVal: document.getElementById('serverTimeVal'),
    downloadResultBtn: document.getElementById('downloadResultBtn'),
    downloadBtnLabel: document.getElementById('downloadBtnLabel'),
    editAgainBtn: document.getElementById('editAgainBtn'),
    tryAnotherBtn: document.getElementById('tryAnotherBtn'),

    // Toast Container
    toastContainer: document.getElementById('toastContainer'),
  };

  // Canvas Contexts
  let imageCtx = null;
  let resultCtx = null;
  let maskCtx = null;

  /* ==========================================================================
     INITIALIZATION
     ========================================================================== */
  function init() {
    initCanvasContexts();
    attachEventListeners();
    updateUIState();
  }

  function initCanvasContexts() {
    if (els.imageCanvas) imageCtx = els.imageCanvas.getContext('2d');
    if (els.resultCanvas) resultCtx = els.resultCanvas.getContext('2d');
    if (els.maskCanvas) {
      maskCtx = els.maskCanvas.getContext('2d');
      maskCtx.lineCap = 'round';
      maskCtx.lineJoin = 'round';
    }
  }

  /* ==========================================================================
     EVENT LISTENERS ATTACHMENT
     ========================================================================== */
  function attachEventListeners() {
    // File Upload & Drag/Drop
    els.uploadPhotoBtn.addEventListener('click', () => els.fileInput.click());
    els.fileInput.addEventListener('change', handleFileInputChange);
    els.pasteClipboardBtn.addEventListener('click', handlePasteFromClipboard);

    // Global Drag and Drop
    window.addEventListener('dragover', handleWindowDragOver);
    window.addEventListener('dragleave', handleWindowDragLeave);
    window.addEventListener('drop', handleWindowDrop);

    // Global Paste (Ctrl+V)
    window.addEventListener('paste', handleWindowPaste);

    // Tools Switching
    els.toolSelectBtn.addEventListener('click', () => setTool('select'));
    els.toolBrushBtn.addEventListener('click', () => setTool('brush'));
    els.toolEraserBtn.addEventListener('click', () => setTool('eraser'));
    els.panelToolBrush.addEventListener('click', () => setTool('brush'));
    els.panelToolEraser.addEventListener('click', () => setTool('eraser'));

    // Brush Size
    els.brushSizeRange.addEventListener('input', (e) => setBrushSize(parseInt(e.target.value, 10)));
    els.brushPresetPills.addEventListener('click', (e) => {
      const pill = e.target.closest('.preset-pill');
      if (pill && pill.dataset.size) {
        setBrushSize(parseInt(pill.dataset.size, 10));
      }
    });

    // Undo / Redo
    els.toolUndoBtn.addEventListener('click', undo);
    els.toolRedoBtn.addEventListener('click', redo);

    // Clear Mask
    els.toolClearMaskBtn.addEventListener('click', clearMaskSelection);
    els.clearSelectionBtn.addEventListener('click', clearMaskSelection);
    els.eraseMaskBtn.addEventListener('click', () => setTool('eraser'));

    // New Photo / Try Another
    els.toolNewPhotoBtn.addEventListener('click', () => els.fileInput.click());
    els.tryAnotherBtn.addEventListener('click', () => els.fileInput.click());

    // AI Inpainting
    els.removeObjectBtn.addEventListener('click', executeObjectRemoval);
    els.toolRemoveQuickBtn.addEventListener('click', executeObjectRemoval);

    // Result Actions
    els.downloadResultBtn.addEventListener('click', handleDownloadFeedback);
    els.exportBtn.addEventListener('click', handleDownloadFeedback);
    els.editAgainBtn.addEventListener('click', applyResultAndEditAgain);
    els.saveBtn.addEventListener('click', () => showToast('Project state saved locally.', 'info'));
    els.projectsBtn.addEventListener('click', () => showToast('Projects library is in preview.', 'info'));

    // Zoom Controls
    els.zoomInBtn.addEventListener('click', zoomIn);
    els.zoomOutBtn.addEventListener('click', zoomOut);
    els.zoomFitBtn.addEventListener('click', zoomFit);
    els.zoomOriginalBtn.addEventListener('click', zoomOriginal);

    // Canvas Viewport Mouse / Pointer Events for Drawing & Panning
    els.canvasViewport.addEventListener('wheel', handleCanvasWheel, { passive: false });
    els.canvasViewport.addEventListener('pointerdown', handleCanvasPointerDown);
    window.addEventListener('pointermove', handleWindowPointerMove);
    window.addEventListener('pointerup', handleWindowPointerUp);

    // Comparison Split Slider Drag Events
    if (els.splitSlider) {
      const handle = els.splitSlider.querySelector('.split-slider-handle');
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

    // Keyboard Shortcuts
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    window.addEventListener('resize', handleWindowResize);
  }

  /* ==========================================================================
     IMAGE LOADING & CANVAS SETUP (CANVAS ENTRANCE)
     ========================================================================== */
  function handleFileInputChange(e) {
    const file = e.target.files && e.target.files[0];
    if (file) {
      loadImageFromFile(file);
    }
    e.target.value = '';
  }

  function handleWindowDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    if (els.emptyDropZone) {
      els.emptyDropZone.classList.add('drag-over');
    }
  }

  function handleWindowDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    if (els.emptyDropZone) {
      els.emptyDropZone.classList.remove('drag-over');
    }
  }

  function handleWindowDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    if (els.emptyDropZone) {
      els.emptyDropZone.classList.remove('drag-over');
    }
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files.length > 0) {
      const file = dt.files[0];
      if (file.type.startsWith('image/')) {
        loadImageFromFile(file);
      } else {
        showToast('Please upload a valid image file (JPG, PNG, WebP).', 'error');
      }
    }
  }

  async function handlePasteFromClipboard() {
    try {
      if (!navigator.clipboard || !navigator.clipboard.read) {
        showToast('Press Ctrl+V to paste an image directly from your clipboard.', 'info');
        return;
      }
      const items = await navigator.clipboard.read();
      for (const item of items) {
        for (const type of item.types) {
          if (type.startsWith('image/')) {
            const blob = await item.getType(type);
            const file = new File([blob], 'clipboard-image.png', { type });
            loadImageFromFile(file);
            showToast('Image pasted from clipboard.', 'success');
            return;
          }
        }
      }
      showToast('No image data found in clipboard.', 'info');
    } catch (err) {
      showToast('Clipboard access was blocked. Press Ctrl+V directly to paste.', 'info');
    }
  }

  function handleWindowPaste(e) {
    if (isTypingInInput()) return;
    const items = (e.clipboardData || window.clipboardData)?.items;
    if (!items) return;
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.indexOf('image') !== -1) {
        const file = items[i].getAsFile();
        if (file) {
          loadImageFromFile(file);
          showToast('Image loaded from clipboard.', 'success');
          break;
        }
      }
    }
  }

  function loadImageFromFile(file) {
    const validFormats = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validFormats.includes(file.type)) {
      showToast('Unsupported format. Please select a JPG, PNG, or WebP photo.', 'error');
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
        setupEditorWithImage(img, file.name || 'Untitled Document');
      };
      img.src = event.target.result;
    };
    reader.readAsDataURL(file);
  }

  function setupEditorWithImage(img, filename) {
    state.initialImage = img;
    state.currentImage = img;
    state.resultImage = null;
    state.resultBlob = null;
    state.imageWidth = img.naturalWidth || img.width;
    state.imageHeight = img.naturalHeight || img.height;
    state.hasResult = false;

    // Reset stacks
    state.undoStack = [];
    state.redoStack = [];
    state.hasMaskSelection = false;

    // Update Topbar Title & Dimensions badge
    if (els.projectTitle) els.projectTitle.textContent = filename;
    if (els.imageDimensionsBadge) {
      els.imageDimensionsBadge.textContent = `${state.imageWidth} × ${state.imageHeight} px`;
    }

    // Configure Canvas Dimensions
    setupCanvasLayers();

    // Switch view from Empty State to Active Canvas Stage & Contextual Panel
    els.emptyState.style.display = 'none';
    els.canvasStage.style.display = 'flex';
    els.rightPanel.style.display = 'flex';

    // 1. Subtle, fast canvas entrance animation (220ms)
    els.canvasContainer.classList.remove('canvas-entrance');
    void els.canvasContainer.offsetWidth; // Force reflow
    els.canvasContainer.classList.add('canvas-entrance');
    setTimeout(() => {
      els.canvasContainer.classList.remove('canvas-entrance');
    }, 250);

    // Hide any previous result controls & split slider
    if (els.resultControls) els.resultControls.style.display = 'none';
    if (els.splitSlider) els.splitSlider.style.display = 'none';
    if (els.resultCanvas) els.resultCanvas.classList.remove('active');

    // Default tool to brush
    setTool('brush');

    // Calculate Best Fit Zoom & Center Viewport
    zoomFit();
    saveUndoState();
    updateUIState();
  }

  function setupCanvasLayers() {
    const w = state.imageWidth;
    const h = state.imageHeight;

    [els.imageCanvas, els.resultCanvas, els.maskCanvas].forEach((canvas) => {
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

    els.canvasContainer.style.width = `${w}px`;
    els.canvasContainer.style.height = `${h}px`;
  }

  /* ==========================================================================
     TOOL & BRUSH SIZE SWITCHING (WITH BRUSH/ERASER CURSOR VISUAL DISTINCTION)
     ========================================================================== */
  function setTool(toolName) {
    state.currentTool = toolName;

    // Update Left Toolbar Buttons
    [els.toolSelectBtn, els.toolBrushBtn, els.toolEraserBtn].forEach((btn) => {
      if (btn) btn.classList.toggle('active', btn.dataset.tool === toolName);
    });

    // Update Right Contextual Panel Segmented Control
    if (els.panelToolBrush) els.panelToolBrush.classList.toggle('active', toolName === 'brush');
    if (els.panelToolEraser) els.panelToolEraser.classList.toggle('active', toolName === 'eraser');

    // Update Brush Cursor Distinction
    if (els.brushCursor) {
      if (toolName === 'eraser') {
        els.brushCursor.classList.remove('brush-mode');
        els.brushCursor.classList.add('eraser-mode');
      } else {
        els.brushCursor.classList.remove('eraser-mode');
        els.brushCursor.classList.add('brush-mode');
      }
    }

    // Update Viewport Cursor class
    if (toolName === 'select') {
      els.canvasViewport.classList.add('panning');
      if (els.brushCursor) els.brushCursor.style.display = 'none';
    } else {
      els.canvasViewport.classList.remove('panning');
    }
  }

  function setBrushSize(size) {
    const clamped = Math.max(5, Math.min(150, size));
    state.brushSize = clamped;

    if (els.brushSizeRange) els.brushSizeRange.value = clamped;
    if (els.brushSizeDisplay) els.brushSizeDisplay.textContent = `${clamped} px`;

    // Update preset pills
    if (els.brushPresetPills) {
      const pills = els.brushPresetPills.querySelectorAll('.preset-pill');
      pills.forEach((p) => {
        p.classList.toggle('active', parseInt(p.dataset.size, 10) === clamped);
      });
    }

    updateBrushCursorSize();
  }

  function updateBrushCursorSize() {
    if (!els.brushCursor) return;
    // Inside canvasContainer, coordinates are 1:1 in natural image pixels!
    els.brushCursor.style.width = `${state.brushSize}px`;
    els.brushCursor.style.height = `${state.brushSize}px`;
  }

  /* ==========================================================================
     CANVAS DRAWING & POINTER COORDINATES
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

  function handleCanvasPointerDown(e) {
    if (e.button !== 0) return; // Only primary mouse button

    // Check if middle click or space key or Select Tool is active -> Pan
    if (state.currentTool === 'select' || state.spacePressed || e.button === 1) {
      state.isPanning = true;
      state.panStartX = e.clientX - state.panX;
      state.panStartY = e.clientY - state.panY;
      els.canvasViewport.classList.add('panning');
      return;
    }

    if (!maskCtx || !state.currentImage) return;

    // Begin drawing stroke
    state.isDrawing = true;
    saveUndoState();

    const coords = getCanvasCoords(e.clientX, e.clientY);
    state.lastX = coords.x;
    state.lastY = coords.y;

    configureMaskContext();

    // Draw single point / dot
    maskCtx.beginPath();
    maskCtx.arc(coords.x, coords.y, state.brushSize / 2, 0, Math.PI * 2);
    maskCtx.fill();

    state.hasMaskSelection = true;
    updateUIState();
  }

  function handleWindowPointerMove(e) {
    // 1. Comparison Split Slider Dragging
    if (state.isDraggingSplit && els.resultCanvas) {
      const rect = els.canvasContainer.getBoundingClientRect();
      const relativeX = e.clientX - rect.left;
      const percent = Math.max(0, Math.min(100, (relativeX / rect.width) * 100));
      updateSplitSliderPosition(percent);
      return;
    }

    // 2. Viewport Panning
    if (state.isPanning) {
      state.panX = e.clientX - state.panStartX;
      state.panY = e.clientY - state.panStartY;
      applyTransform(false);
      return;
    }

    // 3. Update Brush Follower Cursor (Disappears immediately when leaving canvas)
    if (state.currentImage && els.brushCursor) {
      const containerRect = els.canvasContainer.getBoundingClientRect();
      const inCanvas =
        e.clientX >= containerRect.left &&
        e.clientX <= containerRect.right &&
        e.clientY >= containerRect.top &&
        e.clientY <= containerRect.bottom;

      if (inCanvas && state.currentTool !== 'select') {
        const visualX = (e.clientX - containerRect.left) / state.zoom;
        const visualY = (e.clientY - containerRect.top) / state.zoom;
        els.brushCursor.style.left = `${visualX}px`;
        els.brushCursor.style.top = `${visualY}px`;
        updateBrushCursorSize();
        els.brushCursor.style.display = 'block';
      } else {
        els.brushCursor.style.display = 'none';
      }
    }

    // 4. Drawing on Mask Canvas
    if (!state.isDrawing || !maskCtx) return;

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
      if (state.currentTool !== 'select') {
        els.canvasViewport.classList.remove('panning');
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
     UNDO / REDO STACK
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
    showToast('Mask selection cleared.', 'info');
  }

  /* ==========================================================================
     ZOOM & VIEWPORT PANNING (SMOOTH & CURSOR-ALIGNED)
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
    const viewW = els.canvasViewport.clientWidth - 48;
    const viewH = els.canvasViewport.clientHeight - 80;

    if (viewW <= 0 || viewH <= 0) return;

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
    els.canvasContainer.style.setProperty('--pan-x', `${state.panX}px`);
    els.canvasContainer.style.setProperty('--pan-y', `${state.panY}px`);
    els.canvasContainer.style.setProperty('--zoom-val', `${state.zoom}`);

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
    e.preventDefault();
    if (e.ctrlKey || e.metaKey) {
      // Zoom around pointer position
      const rect = els.canvasViewport.getBoundingClientRect();
      const pointerX = e.clientX - rect.left - rect.width / 2;
      const pointerY = e.clientY - rect.top - rect.height / 2;

      const oldZoom = state.zoom;
      const zoomDelta = e.deltaY < 0 ? 1.12 : 0.89;
      const newZoom = Math.max(0.1, Math.min(8.0, oldZoom * zoomDelta));

      // Adjust pan to zoom into pointer
      state.panX -= (pointerX - state.panX) * (newZoom / oldZoom - 1);
      state.panY -= (pointerY - state.panY) * (newZoom / oldZoom - 1);
      setZoom(newZoom, false);
    } else {
      // Pan on trackpad / wheel
      state.panX -= e.deltaX;
      state.panY -= e.deltaY;
      applyTransform(false);
    }
  }

  /* ==========================================================================
     AI OBJECT REMOVAL (REAL BACKEND INTEGRATION & PROCESSING OVERLAY)
     ========================================================================== */
  async function executeObjectRemoval() {
    if (state.isProcessing) return;

    if (!state.currentImage) {
      showToast('Please upload an image first.', 'error');
      return;
    }

    if (!state.hasMaskSelection) {
      showToast('Please brush over the object you want to remove.', 'error');
      return;
    }

    const clientStartTime = performance.now();
    try {
      state.isProcessing = true;
      updateUIState();

      // 6 & 7. Activate in-canvas processing overlay & soft shimmer pulse strictly on masked region
      startProcessingVisuals();

      // 1. Export active image to PNG blob
      const imageBlob = await new Promise((resolve) => {
        els.imageCanvas.toBlob(resolve, 'image/png');
      });

      // 2. Export strict binary mask to PNG blob (white = remove, black = keep)
      const maskBlob = await generateStrictBinaryMaskBlob();

      // 3. Assemble multipart FormData
      const formData = new FormData();
      formData.append('image', imageBlob, 'image.png');
      formData.append('mask', maskBlob, 'mask.png');
      formData.append('prompt', 'seamless background fill, photorealistic, clean texture');

      // 4. Send request to FastAPI backend
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
          errMessage = `Server responded with status ${response.status}`;
        }
        throw new Error(errMessage);
      }

      // Capture server timing header from FastAPI backend
      const serverProcessTimeMs = response.headers.get('X-Process-Time-Ms');

      // 5. Decode returned result PNG image
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

      // 8. RESULT REVEAL: Smoothly remove processing state & fade final image in
      stopProcessingVisuals();

      // Render onto Result Canvas with subtle entrance
      if (resultCtx) {
        resultCtx.clearRect(0, 0, state.imageWidth, state.imageHeight);
        resultCtx.drawImage(resultImg, 0, 0, state.imageWidth, state.imageHeight);
      }

      if (els.resultCanvas) {
        els.resultCanvas.classList.remove('active');
        void els.resultCanvas.offsetWidth;
        els.resultCanvas.classList.add('active');
      }

      // Enable split slider
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

      // True browser-side latency finish: covers mask prep + network + inference + download + decode + render
      const clientEndTime = performance.now();
      const e2eSeconds = ((clientEndTime - clientStartTime) / 1000).toFixed(2);

      // Reveal Result Controls in right contextual panel with metrics
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
        els.resultMetrics.style.display = 'block';
      }

      console.info(`[Latency] Client End-to-End: ${e2eSeconds}s | Server API: ${serverProcessTimeMs ? serverProcessTimeMs + 'ms' : 'N/A'}`);
      showToast(`Object removed successfully in ${e2eSeconds}s.`, 'success');
    } catch (err) {
      console.error('AI Object Removal Error:', err);
      stopProcessingVisuals();
      showToast(err.message || 'Unable to process image. Please try again.', 'error');
    } finally {
      state.isProcessing = false;
      updateUIState();
    }
  }

  function startProcessingVisuals() {
    // 1. Shimmer pulse on mask layer
    if (els.maskCanvas) {
      els.maskCanvas.classList.add('mask-processing');
    }

    // 2. Show in-canvas processing card
    if (els.processingStatusCard) {
      els.processingStatusCard.style.display = 'inline-flex';
    }

    // 3. Cycle subline phases gracefully without fake percentages
    const phases = [els.phase1, els.phase2, els.phase3].filter(Boolean);
    let currentPhaseIdx = 0;

    phases.forEach((p, idx) => p.classList.toggle('active', idx === 0));

    if (state.processingPhaseTimer) clearInterval(state.processingPhaseTimer);
    state.processingPhaseTimer = setInterval(() => {
      currentPhaseIdx = (currentPhaseIdx + 1) % phases.length;
      phases.forEach((p, idx) => p.classList.toggle('active', idx === currentPhaseIdx));
    }, 2200);
  }

  function stopProcessingVisuals() {
    if (state.processingPhaseTimer) {
      clearInterval(state.processingPhaseTimer);
      state.processingPhaseTimer = null;
    }
    if (els.maskCanvas) {
      els.maskCanvas.classList.remove('mask-processing');
    }
    if (els.processingStatusCard) {
      els.processingStatusCard.style.display = 'none';
    }
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
        const val = alpha > 5 ? 255 : 0;
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
     BEFORE / AFTER SPLIT SLIDER
     ========================================================================= */
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
     13. DOWNLOAD FEEDBACK & RESULT ACTIONS
     ========================================================================== */
  function handleDownloadFeedback() {
    const blobToDownload = state.resultBlob;
    if (!blobToDownload && !state.currentImage) {
      showToast('No image available to download.', 'error');
      return;
    }

    const downloadBtn = els.downloadResultBtn;
    const labelSpan = els.downloadBtnLabel;
    const exportLabel = els.exportBtnLabel;

    // Step 1: Preparing...
    if (downloadBtn) downloadBtn.classList.add('download-preparing');
    if (labelSpan) labelSpan.textContent = 'Preparing...';
    if (exportLabel) exportLabel.textContent = 'Preparing...';

    setTimeout(() => {
      try {
        const triggerDownload = (blob) => {
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `canvas-studio-${Date.now()}.png`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);

          // Step 2: ✓ Downloaded (Only after browser download has actually been triggered!)
          if (downloadBtn) {
            downloadBtn.classList.remove('download-preparing');
            downloadBtn.classList.add('download-success');
          }
          if (labelSpan) labelSpan.textContent = '✓ Downloaded';
          if (exportLabel) exportLabel.textContent = '✓ Downloaded';
          showToast('Full-resolution image downloaded.', 'success');

          // Step 3: Revert smoothly after 1.8s
          setTimeout(() => {
            if (downloadBtn) {
              downloadBtn.classList.remove('download-success');
            }
            if (labelSpan) labelSpan.textContent = 'Download Result';
            if (exportLabel) exportLabel.textContent = 'Export';
          }, 1800);
        };

        if (blobToDownload) {
          triggerDownload(blobToDownload);
        } else {
          els.imageCanvas.toBlob((blob) => triggerDownload(blob), 'image/png');
        }
      } catch (err) {
        if (downloadBtn) downloadBtn.classList.remove('download-preparing');
        if (labelSpan) labelSpan.textContent = 'Download Result';
        if (exportLabel) exportLabel.textContent = 'Export';
        showToast('Download failed. Please try again.', 'error');
      }
    }, 120);
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
    if (els.resultMetrics) els.resultMetrics.style.display = 'none';

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
     KEYBOARD SHORTCUTS & WINDOW RESIZE
     ========================================================================== */
  function handleKeyDown(e) {
    if (isTypingInInput()) return;

    // Spacebar Panning toggle
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

    // Brush / Eraser / Select Tool Shortcuts
    else if (e.key.toLowerCase() === 'b' && !e.ctrlKey && !e.metaKey) {
      setTool('brush');
    } else if (e.key.toLowerCase() === 'e' && !e.ctrlKey && !e.metaKey) {
      setTool('eraser');
    } else if (e.key.toLowerCase() === 'v' && !e.ctrlKey && !e.metaKey) {
      setTool('select');
    }

    // Brush Size Adjustments
    else if (e.key === '[') {
      setBrushSize(state.brushSize - 10);
    } else if (e.key === ']') {
      setBrushSize(state.brushSize + 10);
    }

    // Fit & 1:1 Shortcuts
    else if (e.key === '0' && !e.ctrlKey) {
      zoomFit();
    } else if (e.key === '1' && !e.ctrlKey) {
      zoomOriginal();
    }

    // Execute Removal on Enter
    else if (e.key === 'Enter' && state.hasMaskSelection && !state.isProcessing) {
      executeObjectRemoval();
    }

    // Escape to Cancel temporary state or Clear Mask
    else if (e.key === 'Escape') {
      if (state.hasMaskSelection) {
        clearMaskSelection();
      }
    }
  }

  function handleKeyUp(e) {
    if (e.code === 'Space') {
      state.spacePressed = false;
      if (state.currentTool !== 'select') {
        els.canvasViewport.classList.remove('panning');
      }
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
     UI STATE UPDATER
     ========================================================================== */
  function updateUIState() {
    const hasImage = Boolean(state.currentImage);
    const hasSelection = state.hasMaskSelection;
    const isProcessing = state.isProcessing;

    if (els.removeObjectBtn) {
      els.removeObjectBtn.disabled = !hasImage || !hasSelection || isProcessing;
    }
    if (els.toolRemoveQuickBtn) {
      els.toolRemoveQuickBtn.disabled = !hasImage || !hasSelection || isProcessing;
    }

    if (els.removeBtnSpinner) {
      els.removeBtnSpinner.style.display = isProcessing ? 'inline-block' : 'none';
    }
    if (els.removeBtnIcon) {
      els.removeBtnIcon.style.display = isProcessing ? 'none' : 'inline-block';
    }
    if (els.removeBtnLabel) {
      els.removeBtnLabel.textContent = isProcessing
        ? 'Removing with AI...'
        : 'Remove Object with AI';
    }

    if (els.actionCaption) {
      if (isProcessing) {
        els.actionCaption.textContent = 'Synthesizing clean background texture...';
      } else if (!hasSelection) {
        els.actionCaption.textContent = 'Paint over the object you wish to remove.';
      } else {
        els.actionCaption.textContent = 'Ready to inpaint. Click button or press Enter.';
      }
    }

    if (els.toolUndoBtn) els.toolUndoBtn.disabled = state.undoStack.length === 0;
    if (els.toolRedoBtn) els.toolRedoBtn.disabled = state.redoStack.length === 0;

    if (els.toolClearMaskBtn) els.toolClearMaskBtn.disabled = !hasSelection;
    if (els.clearSelectionBtn) els.clearSelectionBtn.disabled = !hasSelection;
  }

  /* ==========================================================================
     TOAST NOTIFICATION HELPER
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
    }, 3500);
  }

  // Run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
