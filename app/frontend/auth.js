/**
 * AI Photo Studio — Authentication Service & Modal Controller
 * Supports Supabase Auth (Email/Password, Google OAuth, Password Reset, Persistent Sessions)
 * with graceful local authentication fallback when Supabase credentials are not configured.
 */

(function () {
  'use strict';

  // Storage keys for local fallback
  const STORAGE_SESSION_KEY = 'ai_photo_studio_auth_session';
  const STORAGE_USERS_KEY = 'ai_photo_studio_users';

  class AuthenticationService {
    constructor() {
      this.supabaseClient = null;
      this.isConfigured = false;
      this.currentUser = null;
      this.authListeners = [];
      this.currentView = 'login';
      this.isProcessing = false;

      // Pending action to execute after successful login (e.g., file upload)
      this.pendingAction = null;
    }

    /**
     * Initializes the auth service: fetches config from backend,
     * initializes Supabase or local mode, restores session, and binds modal events.
     */
    async init() {
      this.bindModalElements();
      await this.loadConfigAndSession();
      this.bindEvents();
      this.updateNavbarUI();
    }

    /**
     * Fetch auth configuration from backend
     */
    async loadConfigAndSession() {
      let config = { supabase_url: '', supabase_anon_key: '' };
      try {
        const res = await fetch('/api/auth/config');
        if (res.ok) {
          config = await res.json();
        }
      } catch (err) {
        console.warn('Could not fetch auth config from /api/auth/config, checking window globals:', err);
      }

      const url = (config.supabase_url || window.SUPABASE_URL || '').trim();
      const key = (config.supabase_anon_key || window.SUPABASE_ANON_KEY || '').trim();

      // Check if credentials are real and not placeholder
      const isPlaceholder = !url || !key || 
        url.includes('your_supabase') || 
        key.includes('your_supabase') || 
        !url.startsWith('http');

      if (!isPlaceholder && window.supabase && typeof window.supabase.createClient === 'function') {
        try {
          this.supabaseClient = window.supabase.createClient(url, key, {
            auth: {
              persistSession: true,
              autoRefreshToken: true,
              detectSessionInUrl: true,
            },
          });
          this.isConfigured = true;

          // Check current active Supabase session
          const { data, error } = await this.supabaseClient.auth.getSession();
          if (data && data.session && data.session.user) {
            this.currentUser = this.formatUser(data.session.user);
          }

          // Listen to live Supabase auth state changes
          this.supabaseClient.auth.onAuthStateChange((event, session) => {
            if (session && session.user) {
              this.currentUser = this.formatUser(session.user);
            } else {
              this.currentUser = null;
            }
            this.notifyListeners();
            this.updateNavbarUI();
          });

          this.updateModeNotice(true);
        } catch (e) {
          console.error('Failed to initialize Supabase client:', e);
          this.initLocalFallback();
        }
      } else {
        this.initLocalFallback();
      }

      this.notifyListeners();
      this.updateNavbarUI();
    }

    /**
     * Local storage session manager for demo / offline use
     */
    initLocalFallback() {
      this.isConfigured = false;
      this.updateModeNotice(false);

      try {
        const stored = localStorage.getItem(STORAGE_SESSION_KEY);
        if (stored) {
          this.currentUser = JSON.parse(stored);
        }
      } catch (_) {
        this.currentUser = null;
      }
    }

    updateModeNotice(isLiveSupabase) {
      const hint = document.getElementById('authModeHint');
      if (!hint) return;
      if (isLiveSupabase) {
        hint.innerHTML = '<span class="status-dot green"></span> Connected to Supabase Cloud Auth';
        hint.className = 'auth-mode-hint connected';
      } else {
        hint.innerHTML = '<span class="status-dot amber"></span> Local Auth Mode (Set SUPABASE_URL &amp; SUPABASE_ANON_KEY in .env for production)';
        hint.className = 'auth-mode-hint local';
      }
    }

    formatUser(user) {
      if (!user) return null;
      const email = user.email || 'User';
      const name = user.user_metadata?.full_name || user.user_metadata?.name || email.split('@')[0];
      return {
        id: user.id || 'usr_' + Date.now(),
        email: email,
        name: name,
        initial: (name || email || 'U').charAt(0).toUpperCase(),
        provider: user.app_metadata?.provider || 'email',
      };
    }

    /**
     * Returns true if user is logged in
     */
    isAuthenticated() {
      return !!this.currentUser;
    }

    /**
     * Returns current user object or null
     */
    getUser() {
      return this.currentUser;
    }

    /**
     * Register listener for auth state change
     */
    onAuthStateChange(callback) {
      if (typeof callback === 'function') {
        this.authListeners.push(callback);
        // Immediately fire current state
        callback(this.currentUser);
      }
    }

    notifyListeners() {
      this.authListeners.forEach((cb) => {
        try {
          cb(this.currentUser);
        } catch (e) {
          console.error('Error in auth listener:', e);
        }
      });
    }

    /* ==========================================================================
       AUTHENTICATION ACTIONS
       ========================================================================== */

    async login(email, password) {
      if (this.isConfigured && this.supabaseClient) {
        const { data, error } = await this.supabaseClient.auth.signInWithPassword({
          email: email.trim(),
          password: password,
        });
        if (error) {
          throw new Error(error.message || 'Invalid email or password.');
        }
        this.currentUser = this.formatUser(data.user);
        this.notifyListeners();
        this.updateNavbarUI();
        return this.currentUser;
      }

      // Local fallback mode
      await new Promise((r) => setTimeout(r, 450)); // simulate network delay
      const users = this.getLocalUsers();
      const existing = users.find((u) => u.email.toLowerCase() === email.trim().toLowerCase());

      if (existing) {
        if (existing.password !== password) {
          throw new Error('Incorrect password. Please try again.');
        }
        this.currentUser = this.formatUser(existing);
      } else {
        // If not in local registered list, allow standard demo signin
        const newUser = {
          id: 'local_' + Date.now(),
          email: email.trim(),
          password: password,
          user_metadata: { full_name: email.split('@')[0] },
        };
        users.push(newUser);
        this.saveLocalUsers(users);
        this.currentUser = this.formatUser(newUser);
      }

      localStorage.setItem(STORAGE_SESSION_KEY, JSON.stringify(this.currentUser));
      this.notifyListeners();
      this.updateNavbarUI();
      return this.currentUser;
    }

    async signup(email, password) {
      if (this.isConfigured && this.supabaseClient) {
        const { data, error } = await this.supabaseClient.auth.signUp({
          email: email.trim(),
          password: password,
          options: {
            data: { full_name: email.split('@')[0] },
          },
        });
        if (error) {
          throw new Error(error.message || 'Signup failed. Please try again.');
        }

        // Check if email confirmation is required by Supabase
        if (data.user && (!data.session || data.user.identities?.length === 0)) {
          return {
            requiresConfirmation: true,
            user: this.formatUser(data.user),
            message: 'Account created! Please check your email to confirm your account.',
          };
        }

        this.currentUser = this.formatUser(data.user);
        this.notifyListeners();
        this.updateNavbarUI();
        return { requiresConfirmation: false, user: this.currentUser };
      }

      // Local fallback mode
      await new Promise((r) => setTimeout(r, 450));
      const users = this.getLocalUsers();
      const existing = users.find((u) => u.email.toLowerCase() === email.trim().toLowerCase());
      if (existing) {
        throw new Error('An account with this email already exists. Please log in.');
      }

      const newUser = {
        id: 'local_' + Date.now(),
        email: email.trim(),
        password: password,
        user_metadata: { full_name: email.split('@')[0] },
      };
      users.push(newUser);
      this.saveLocalUsers(users);

      this.currentUser = this.formatUser(newUser);
      localStorage.setItem(STORAGE_SESSION_KEY, JSON.stringify(this.currentUser));
      this.notifyListeners();
      this.updateNavbarUI();
      return { requiresConfirmation: false, user: this.currentUser };
    }

    async loginWithGoogle() {
      if (this.isConfigured && this.supabaseClient) {
        const { data, error } = await this.supabaseClient.auth.signInWithOAuth({
          provider: 'google',
          options: {
            redirectTo: window.location.origin + window.location.pathname,
          },
        });
        if (error) {
          throw new Error(error.message || 'Google OAuth failed.');
        }
        return data;
      }

      // Local fallback mode
      await new Promise((r) => setTimeout(r, 500));
      const mockGoogleUser = {
        id: 'google_' + Date.now(),
        email: 'alex.google@example.com',
        user_metadata: { full_name: 'Alex Rivera' },
        app_metadata: { provider: 'google' },
      };
      this.currentUser = this.formatUser(mockGoogleUser);
      localStorage.setItem(STORAGE_SESSION_KEY, JSON.stringify(this.currentUser));
      this.notifyListeners();
      this.updateNavbarUI();
      return { user: this.currentUser };
    }

    async resetPassword(email) {
      if (this.isConfigured && this.supabaseClient) {
        const { error } = await this.supabaseClient.auth.resetPasswordForEmail(email.trim(), {
          redirectTo: window.location.origin + window.location.pathname,
        });
        if (error) {
          throw new Error(error.message || 'Password reset request failed.');
        }
        return true;
      }

      // Local fallback mode
      await new Promise((r) => setTimeout(r, 400));
      return true;
    }

    async logout() {
      if (this.isConfigured && this.supabaseClient) {
        try {
          await this.supabaseClient.auth.signOut();
        } catch (e) {
          console.warn('Supabase signOut error:', e);
        }
      }

      this.currentUser = null;
      try {
        localStorage.removeItem(STORAGE_SESSION_KEY);
      } catch (_) {}

      this.notifyListeners();
      this.updateNavbarUI();
    }

    getLocalUsers() {
      try {
        const raw = localStorage.getItem(STORAGE_USERS_KEY);
        return raw ? JSON.parse(raw) : [];
      } catch (_) {
        return [];
      }
    }

    saveLocalUsers(users) {
      try {
        localStorage.setItem(STORAGE_USERS_KEY, JSON.stringify(users));
      } catch (_) {}
    }

    /* ==========================================================================
       NAVBAR UI UPDATE
       ========================================================================== */

    updateNavbarUI() {
      const loginBtn = document.getElementById('navLoginBtn');
      const userProfile = document.getElementById('navUserProfile');
      const userEmail = document.getElementById('userEmail');
      const userAvatar = document.getElementById('userAvatar');

      if (!loginBtn || !userProfile) return;

      if (this.currentUser) {
        loginBtn.style.display = 'none';
        userProfile.style.display = 'flex';
        if (userEmail) userEmail.textContent = this.currentUser.email;
        if (userAvatar) userAvatar.textContent = this.currentUser.initial;
      } else {
        loginBtn.style.display = 'inline-flex';
        userProfile.style.display = 'none';
      }
    }

    /* ==========================================================================
       MODAL CONTROLLER & VALIDATIONS
       ========================================================================== */

    bindModalElements() {
      this.dom = {
        overlay: document.getElementById('authModalOverlay'),
        card: document.getElementById('authModalCard'),
        closeBtn: document.getElementById('authModalClose'),
        alertBox: document.getElementById('authAlert'),

        // Views
        viewLogin: document.getElementById('viewLogin'),
        viewSignup: document.getElementById('viewSignup'),
        viewForgot: document.getElementById('viewForgotPassword'),

        // Login inputs & buttons
        loginForm: document.getElementById('loginForm'),
        loginEmail: document.getElementById('loginEmail'),
        loginPassword: document.getElementById('loginPassword'),
        loginEmailError: document.getElementById('loginEmailError'),
        loginPasswordError: document.getElementById('loginPasswordError'),
        btnLoginSubmit: document.getElementById('btnLoginSubmit'),
        btnGoogleLogin: document.getElementById('btnGoogleLogin'),
        linkForgotPassword: document.getElementById('linkForgotPassword'),
        linkGoToSignup: document.getElementById('linkGoToSignup'),

        // Signup inputs & buttons
        signupForm: document.getElementById('signupForm'),
        signupEmail: document.getElementById('signupEmail'),
        signupPassword: document.getElementById('signupPassword'),
        signupPasswordConfirm: document.getElementById('signupPasswordConfirm'),
        signupEmailError: document.getElementById('signupEmailError'),
        signupPasswordError: document.getElementById('signupPasswordError'),
        signupPasswordConfirmError: document.getElementById('signupPasswordConfirmError'),
        btnSignupSubmit: document.getElementById('btnSignupSubmit'),
        btnGoogleSignup: document.getElementById('btnGoogleSignup'),
        linkGoToLogin: document.getElementById('linkGoToLogin'),

        // Forgot password
        forgotPasswordForm: document.getElementById('forgotPasswordForm'),
        resetEmail: document.getElementById('resetEmail'),
        resetEmailError: document.getElementById('resetEmailError'),
        btnResetSubmit: document.getElementById('btnResetSubmit'),
        linkBackToLogin: document.getElementById('linkBackToLogin'),

        // Navbar logout
        navLogoutBtn: document.getElementById('navLogoutBtn'),
      };
    }

    bindEvents() {
      if (!this.dom.overlay) return;

      // Close modal on close button or backdrop click
      if (this.dom.closeBtn) {
        this.dom.closeBtn.addEventListener('click', () => this.closeModal());
      }
      this.dom.overlay.addEventListener('click', (e) => {
        if (e.target === this.dom.overlay) {
          this.closeModal();
        }
      });

      // Escape key closes modal
      window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && this.dom.overlay.classList.contains('active')) {
          this.closeModal();
        }
      });

      // Show/Hide password toggles
      document.querySelectorAll('.btn-toggle-password').forEach((btn) => {
        btn.addEventListener('click', () => {
          const targetId = btn.getAttribute('data-target');
          const input = document.getElementById(targetId);
          if (!input) return;

          const eyeOpen = btn.querySelector('.eye-open');
          const eyeClosed = btn.querySelector('.eye-closed');

          if (input.type === 'password') {
            input.type = 'text';
            if (eyeOpen) eyeOpen.style.display = 'none';
            if (eyeClosed) eyeClosed.style.display = 'inline-block';
            btn.setAttribute('aria-label', 'Hide password');
          } else {
            input.type = 'password';
            if (eyeOpen) eyeOpen.style.display = 'inline-block';
            if (eyeClosed) eyeClosed.style.display = 'none';
            btn.setAttribute('aria-label', 'Show password');
          }
        });
      });

      // View switching
      if (this.dom.linkForgotPassword) {
        this.dom.linkForgotPassword.addEventListener('click', () => this.switchView('forgot'));
      }
      if (this.dom.linkGoToSignup) {
        this.dom.linkGoToSignup.addEventListener('click', () => this.switchView('signup'));
      }
      if (this.dom.linkGoToLogin) {
        this.dom.linkGoToLogin.addEventListener('click', () => this.switchView('login'));
      }
      if (this.dom.linkBackToLogin) {
        this.dom.linkBackToLogin.addEventListener('click', () => this.switchView('login'));
      }

      // Input change clears validation errors
      ['loginEmail', 'loginPassword', 'signupEmail', 'signupPassword', 'signupPasswordConfirm', 'resetEmail'].forEach(
        (id) => {
          const el = document.getElementById(id);
          if (el) {
            el.addEventListener('input', () => {
              el.classList.remove('input-error');
              const err = document.getElementById(`${id}Error`);
              if (err) err.textContent = '';
              this.hideAlert();
            });
          }
        }
      );

      // Form Submissions
      if (this.dom.loginForm) {
        this.dom.loginForm.addEventListener('submit', (e) => this.handleLoginSubmit(e));
      }
      if (this.dom.signupForm) {
        this.dom.signupForm.addEventListener('submit', (e) => this.handleSignupSubmit(e));
      }
      if (this.dom.forgotPasswordForm) {
        this.dom.forgotPasswordForm.addEventListener('submit', (e) => this.handleForgotSubmit(e));
      }

      // Google OAuth Buttons
      if (this.dom.btnGoogleLogin) {
        this.dom.btnGoogleLogin.addEventListener('click', () => this.handleGoogleOAuth());
      }
      if (this.dom.btnGoogleSignup) {
        this.dom.btnGoogleSignup.addEventListener('click', () => this.handleGoogleOAuth());
      }

      // Navbar Logout button
      if (this.dom.navLogoutBtn) {
        this.dom.navLogoutBtn.addEventListener('click', () => this.handleLogout());
      }
    }

    openModal(view = 'login', customAlert = null, pendingAction = null) {
      this.pendingAction = pendingAction || null;
      this.switchView(view);
      this.clearAllErrors();

      if (customAlert) {
        this.showAlert(customAlert, 'info');
      } else {
        this.hideAlert();
      }

      if (this.dom.overlay) {
        this.dom.overlay.classList.add('active');
        this.dom.overlay.setAttribute('aria-hidden', 'false');
      }

      // Auto-focus first input
      setTimeout(() => {
        if (view === 'login' && this.dom.loginEmail) this.dom.loginEmail.focus();
        else if (view === 'signup' && this.dom.signupEmail) this.dom.signupEmail.focus();
        else if (view === 'forgot' && this.dom.resetEmail) this.dom.resetEmail.focus();
      }, 100);
    }

    closeModal() {
      if (this.dom.overlay) {
        this.dom.overlay.classList.remove('active');
        this.dom.overlay.setAttribute('aria-hidden', 'true');
      }
      this.clearAllErrors();
      this.hideAlert();
    }

    switchView(view) {
      this.currentView = view;
      this.clearAllErrors();
      this.hideAlert();

      if (this.dom.viewLogin) this.dom.viewLogin.style.display = view === 'login' ? 'block' : 'none';
      if (this.dom.viewSignup) this.dom.viewSignup.style.display = view === 'signup' ? 'block' : 'none';
      if (this.dom.viewForgot) this.dom.viewForgot.style.display = view === 'forgot' ? 'block' : 'none';
    }

    /* ==========================================================================
       SUBMISSION HANDLERS
       ========================================================================== */

    async handleLoginSubmit(e) {
      e.preventDefault();
      if (this.isProcessing) return;

      const email = this.dom.loginEmail.value.trim();
      const password = this.dom.loginPassword.value;

      let valid = true;
      this.clearAllErrors();

      if (!email) {
        this.setFieldError('loginEmail', 'Please enter your email address.');
        valid = false;
      } else if (!this.isValidEmail(email)) {
        this.setFieldError('loginEmail', 'Please enter a valid email address.');
        valid = false;
      }

      if (!password) {
        this.setFieldError('loginPassword', 'Please enter your password.');
        valid = false;
      }

      if (!valid) return;

      try {
        this.setSubmitting(this.dom.btnLoginSubmit, true, 'Logging in...');
        const user = await this.login(email, password);

        this.closeModal();
        this.showToastNotification(`Welcome back, ${user.name || user.email}!`, 'success');

        // Execute pending action if user tried an action while logged out
        this.executePendingAction();
      } catch (err) {
        this.showAlert(err.message || 'Login failed. Please check your credentials.', 'error');
      } finally {
        this.setSubmitting(this.dom.btnLoginSubmit, false, 'Log in');
      }
    }

    async handleSignupSubmit(e) {
      e.preventDefault();
      if (this.isProcessing) return;

      const email = this.dom.signupEmail.value.trim();
      const password = this.dom.signupPassword.value;
      const confirmPassword = this.dom.signupPasswordConfirm.value;

      let valid = true;
      this.clearAllErrors();

      if (!email) {
        this.setFieldError('signupEmail', 'Please enter your email address.');
        valid = false;
      } else if (!this.isValidEmail(email)) {
        this.setFieldError('signupEmail', 'Please enter a valid email address.');
        valid = false;
      }

      if (!password) {
        this.setFieldError('signupPassword', 'Please enter a password.');
        valid = false;
      } else if (password.length < 6) {
        this.setFieldError('signupPassword', 'Password must be at least 6 characters.');
        valid = false;
      }

      if (!confirmPassword) {
        this.setFieldError('signupPasswordConfirm', 'Please confirm your password.');
        valid = false;
      } else if (password !== confirmPassword) {
        this.setFieldError('signupPasswordConfirm', 'Passwords do not match.');
        valid = false;
      }

      if (!valid) return;

      try {
        this.setSubmitting(this.dom.btnSignupSubmit, true, 'Creating account...');
        const result = await this.signup(email, password);

        if (result.requiresConfirmation) {
          this.showAlert(result.message, 'success');
        } else {
          this.closeModal();
          this.showToastNotification(`Account created! Welcome, ${result.user.name || result.user.email}.`, 'success');
          this.executePendingAction();
        }
      } catch (err) {
        this.showAlert(err.message || 'Signup failed. Please try again.', 'error');
      } finally {
        this.setSubmitting(this.dom.btnSignupSubmit, false, 'Create account');
      }
    }

    async handleForgotSubmit(e) {
      e.preventDefault();
      if (this.isProcessing) return;

      const email = this.dom.resetEmail.value.trim();
      if (!email) {
        this.setFieldError('resetEmail', 'Please enter your email address.');
        return;
      }
      if (!this.isValidEmail(email)) {
        this.setFieldError('resetEmail', 'Please enter a valid email address.');
        return;
      }

      try {
        this.setSubmitting(this.dom.btnResetSubmit, true, 'Sending link...');
        await this.resetPassword(email);
        this.showAlert(
          `Password reset link sent to ${email}. Please check your inbox (and spam folder).`,
          'success'
        );
        this.dom.resetEmail.value = '';
      } catch (err) {
        this.showAlert(err.message || 'Could not send reset link. Please try again.', 'error');
      } finally {
        this.setSubmitting(this.dom.btnResetSubmit, false, 'Send reset link');
      }
    }

    async handleGoogleOAuth() {
      try {
        this.showAlert('Connecting to Google...', 'info');
        const res = await this.loginWithGoogle();
        if (res && res.user) {
          this.closeModal();
          this.showToastNotification(`Signed in with Google as ${res.user.email}!`, 'success');
          this.executePendingAction();
        }
      } catch (err) {
        this.showAlert(err.message || 'Google authentication failed.', 'error');
      }
    }

    async handleLogout() {
      await this.logout();
      this.showToastNotification('You have been logged out.', 'info');
    }

    executePendingAction() {
      if (typeof this.pendingAction === 'function') {
        const action = this.pendingAction;
        this.pendingAction = null;
        try {
          action();
        } catch (e) {
          console.error('Error executing pending action:', e);
        }
      }
    }

    /* ==========================================================================
       HELPERS & UI UTILITIES
       ========================================================================== */

    isValidEmail(email) {
      return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
    }

    setFieldError(fieldId, message) {
      const input = document.getElementById(fieldId);
      const errorSpan = document.getElementById(`${fieldId}Error`);
      if (input) input.classList.add('input-error');
      if (errorSpan) errorSpan.textContent = message;
    }

    clearAllErrors() {
      document.querySelectorAll('.form-input').forEach((input) => input.classList.remove('input-error'));
      document.querySelectorAll('.field-error').forEach((span) => (span.textContent = ''));
    }

    showAlert(message, type = 'error') {
      if (!this.dom.alertBox) return;
      this.dom.alertBox.textContent = message;
      this.dom.alertBox.className = `auth-alert alert-${type}`;
      this.dom.alertBox.style.display = 'block';
    }

    hideAlert() {
      if (!this.dom.alertBox) return;
      this.dom.alertBox.style.display = 'none';
      this.dom.alertBox.textContent = '';
    }

    setSubmitting(btn, isSubmitting, text) {
      this.isProcessing = isSubmitting;
      if (!btn) return;
      btn.disabled = isSubmitting;
      const textSpan = btn.querySelector('.btn-text');
      const spinner = btn.querySelector('.btn-spinner');
      if (textSpan) textSpan.textContent = text;
      if (spinner) spinner.style.display = isSubmitting ? 'inline-block' : 'none';
    }

    showToastNotification(message, type = 'info') {
      // Use existing showToast function if defined in main script, or create minimal fallback
      if (typeof window.showToast === 'function') {
        window.showToast(message, type);
      } else {
        const container = document.getElementById('toastContainer');
        if (!container) return;
        const toast = document.createElement('div');
        toast.className = `studio-toast toast-${type}`;
        toast.textContent = message;
        container.appendChild(toast);
        setTimeout(() => toast.classList.add('toast-visible'), 10);
        setTimeout(() => {
          toast.classList.remove('toast-visible');
          setTimeout(() => toast.remove(), 250);
        }, 3200);
      }
    }
  }

  // Create and expose global singleton instance
  window.AuthService = new AuthenticationService();

  // Initialize on DOMContentLoaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => window.AuthService.init());
  } else {
    window.AuthService.init();
  }
})();
