/**
 * File: frontend/js/main.js
 * Main JavaScript file with common functionality for the MathemAI application
 */

// Constants
const API_BASE_URL = 'http://localhost:5000';

// Check if we're in development or production environment
const isDevelopment = window.location.hostname === 'localhost' || 
                     window.location.hostname === '127.0.0.1';

// Set up environment-specific configurations
if (isDevelopment) {
    console.log('Running in development mode');
} else {
    console.log('Running in production mode');
    // In production, the API URL might be different
    // API_BASE_URL = 'https://api.mathemat.example.com';
}

// Common utility functions

/**
 * Shows a notification to the user
 * @param {string} message - The message to display
 * @param {string} type - The type of notification ('success', 'error', 'info', 'warning')
 * @param {number} duration - How long to show the notification in ms
 */
function showNotification(message, type = 'info', duration = 3000) {
    // Check if notification container exists, create if not
    let container = document.getElementById('notification-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'notification-container';
        container.style.position = 'fixed';
        container.style.top = '20px';
        container.style.right = '20px';
        container.style.zIndex = '1000';
        document.body.appendChild(container);
    }

    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} notification`;
    notification.innerHTML = message;
    notification.style.marginBottom = '10px';
    notification.style.minWidth = '250px';
    
    // Add dismiss button
    const dismissBtn = document.createElement('button');
    dismissBtn.type = 'button';
    dismissBtn.className = 'btn-close';
    dismissBtn.setAttribute('aria-label', 'Close');
    dismissBtn.onclick = function() {
        container.removeChild(notification);
    };
    notification.appendChild(dismissBtn);
    
    // Add to container
    container.appendChild(notification);
    
    // Auto dismiss after duration
    setTimeout(() => {
        if (notification.parentNode === container) {
            container.removeChild(notification);
        }
    }, duration);
}

/**
 * Format date to a readable string
 * @param {Date|string} date - Date object or date string
 * @return {string} Formatted date string
 */
function formatDate(date) {
    if (typeof date === 'string') {
        date = new Date(date);
    }
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

/**
 * Saves user preferences to localStorage
 * @param {Object} preferences - Object containing user preferences
 */
function saveUserPreferences(preferences) {
    localStorage.setItem('mathemat_preferences', JSON.stringify(preferences));
}

/**
 * Loads user preferences from localStorage
 * @return {Object} User preferences object
 */
function loadUserPreferences() {
    const preferences = localStorage.getItem('mathemat_preferences');
    return preferences ? JSON.parse(preferences) : {};
}

/**
 * Checks if the user is authenticated
 * @return {boolean} Whether the user is authenticated
 */
function isAuthenticated() {
    return !!localStorage.getItem('mathemat_auth_token');
}

/**
 * Redirects unauthenticated users to the login page
 * @param {Array} excludedPaths - Array of paths that don't require authentication
 */
function requireAuthentication(excludedPaths = ['index.html', 'login.html']) {
    if (!isAuthenticated()) {
        const currentPath = window.location.pathname.split('/').pop();
        if (!excludedPaths.includes(currentPath)) {
            window.location.href = 'login.html';
        }
    }
}

/**
 * Apply theme preferences
 */
function applyTheme() {
    const preferences = loadUserPreferences();
    if (preferences.darkMode) {
        document.body.classList.add('dark-mode');
    } else {
        document.body.classList.remove('dark-mode');
    }
    
    if (preferences.highContrast) {
        document.body.classList.add('high-contrast');
    } else {
        document.body.classList.remove('high-contrast');
    }
    
    if (preferences.largeText) {
        document.body.classList.add('large-text');
    } else {
        document.body.classList.remove('large-text');
    }
}

// Event listeners for common elements

// Toggle dark mode
document.addEventListener('DOMContentLoaded', () => {
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    if (darkModeToggle) {
        darkModeToggle.addEventListener('click', () => {
            const preferences = loadUserPreferences();
            preferences.darkMode = !preferences.darkMode;
            saveUserPreferences(preferences);
            applyTheme();
        });
    }
    
    // Apply theme preferences on page load
    applyTheme();
    
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
});