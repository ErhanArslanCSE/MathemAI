/**
 * File: frontend/js/api.js
 * API communication functions for MathemAI
 */

/**
 * Handles API requests to the MathemAI backend
 */
class MathemAIApi {
    constructor() {
        // Base URL for API endpoints
        this.baseUrl = 'http://localhost:5000';
        
        // Default headers for all requests
        this.defaultHeaders = {
            'Content-Type': 'application/json'
        };
        
        // Add auth token if available
        const authToken = localStorage.getItem('mathemat_auth_token');
        if (authToken) {
            this.defaultHeaders['Authorization'] = `Bearer ${authToken}`;
        }
    }
    
    /**
     * Makes a request to the API
     * @param {string} endpoint - API endpoint
     * @param {string} method - HTTP method
     * @param {Object} data - Request payload
     * @param {Object} additionalHeaders - Additional headers
     * @return {Promise} Promise resolving to response data
     */
    async request(endpoint, method = 'GET', data = null, additionalHeaders = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        
        const headers = {
            ...this.defaultHeaders,
            ...additionalHeaders
        };
        
        const options = {
            method,
            headers,
            mode: 'cors'
        };
        
        if (data && (method === 'POST' || method === 'PUT' || method === 'PATCH')) {
            options.body = JSON.stringify(data);
        }
        
        try {
            const response = await fetch(url, options);
            
            // Handle response
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status} ${response.statusText}`);
            }
            
            // Parse JSON response
            const responseData = await response.json();
            return responseData;
        } catch (error) {
            console.error('API request error:', error);
            // Log the error to the server
            this.logError(error.message, url, method);
            throw error;
        }
    }
    
    /**
     * Logs an error to the server
     * @param {string} message - Error message
     * @param {string} url - URL that caused the error
     * @param {string} method - HTTP method used
     */
    async logError(message, url, method) {
        try {
            await fetch(`${this.baseUrl}/api/log-error`, {
                method: 'POST',
                headers: this.defaultHeaders,
                body: JSON.stringify({
                    message,
                    url,
                    method,
                    browser: navigator.userAgent,
                    timestamp: new Date().toISOString()
                })
            });
        } catch (error) {
            console.error('Failed to log error:', error);
        }
    }
    
    /**
     * Submits assessment data for dyscalculia screening
     * @param {Object} assessmentData - Student assessment data
     * @return {Promise} Promise resolving to screening results
     */
    async screenForDyscalculia(assessmentData) {
        return this.request('/api/screen', 'POST', assessmentData);
    }
    
    /**
     * Gets intervention recommendations based on assessment data
     * @param {Object} assessmentData - Student assessment data
     * @return {Promise} Promise resolving to intervention recommendations
     */
    async getInterventionRecommendations(assessmentData) {
        return this.request('/api/recommend', 'POST', assessmentData);
    }
    
    /**
     * Saves a new assessment record
     * @param {Object} assessmentData - Assessment data to save
     * @return {Promise} Promise resolving to the saved record
     */
    async saveAssessment(assessmentData) {
        return this.request('/api/save-assessment', 'POST', assessmentData);
    }
    
    /**
     * Saves intervention tracking data
     * @param {Object} interventionData - Intervention data to save
     * @return {Promise} Promise resolving to the saved record
     */
    async saveIntervention(interventionData) {
        return this.request('/api/save-intervention', 'POST', interventionData);
    }
    
    /**
     * Saves error pattern data
     * @param {Object} errorData - Error pattern data to save
     * @return {Promise} Promise resolving to the saved record
     */
    async saveErrorPattern(errorData) {
        return this.request('/api/error-patterns', 'POST', errorData);
    }
    
    /**
     * Gets statistics about the collected data
     * @return {Promise} Promise resolving to statistics data
     */
    async getStatistics() {
        return this.request('/api/stats', 'GET');
    }
    
    /**
     * Checks API health status
     * @return {Promise} Promise resolving to health status
     */
    async checkHealth() {
        return this.request('/health', 'GET');
    }
}

// Create a singleton instance
const api = new MathemAIApi();

// Export the API instance
window.mathematApi = api;