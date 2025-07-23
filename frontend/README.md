# MathemAI Frontend

This directory contains the frontend implementation for the MathemAI project, providing a simple, user-friendly interface to interact with the MathemAI API.

## Overview

The MathemAI frontend consists of three main components:

1. **Assessment Interface**: For collecting student data and screening for dyscalculia
2. **Results Dashboard**: For viewing screening results and recommendations
3. **Progress Tracker**: For monitoring student progress over time

## Technology Stack

- HTML5/CSS3/JavaScript (ES6+)
- Bootstrap 5.3 for responsive design
- Chart.js for data visualization
- Fetch API for making API requests

## Directory Structure

```
frontend/
├── index.html                 # Landing page
├── css/
│   ├── styles.css             # Custom styles
│   └── bootstrap.min.css      # Bootstrap CSS
├── js/
│   ├── main.js                # Main JavaScript file
│   ├── assessment.js          # Assessment functionality
│   ├── results.js             # Results display functionality
│   ├── progress.js            # Progress tracking functionality
│   └── api.js                 # API communication
├── templates/
│   ├── assessment.html        # Assessment page
│   ├── results.html           # Results page
│   └── progress.html          # Progress tracking page
└── assets/
    └── images/                # Images and icons
```

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/openimpactai/OpenImpactAI.git
   cd OpenImpactAI/AI-Education-Projects/MathemAI/frontend
   ```

2. **Open in Browser**:
   You can simply open `index.html` in a web browser to view the frontend.

3. **For Development**:
   For easier development, you can use a simple local server:
   ```bash
   # Using Python
   python -m http.server 8080
   
   # Using Node.js
   npx serve
   ```

## Features

### Assessment Interface
- Multi-step form for collecting student information
- Interactive math skill assessment
- Real-time feedback during assessment
- Simple, engaging UI designed for all ages

### Results Dashboard
- Clear presentation of screening results
- Visualization of strengths and weaknesses
- Personalized intervention recommendations
- Printable/shareable results report

### Progress Tracker
- Visualization of progress over time
- Comparison of pre/post intervention performance
- Goal setting and milestone tracking
- Activity history log

## API Integration

The frontend communicates with the MathemAI API through the functions in `api.js`. Key endpoints used include:

- `/api/screen`: For submitting assessment data and receiving screening results
- `/api/recommend`: For getting personalized intervention recommendations
- `/api/save-assessment`: For saving assessment data
- `/api/save-intervention`: For tracking intervention effectiveness

## Contributing

We welcome contributions to improve the frontend. Please consider:

1. **Accessibility**: Ensure all features are accessible to users with disabilities
2. **Simplicity**: Keep the interface simple and intuitive
3. **Responsiveness**: Ensure the interface works well on all devices
4. **Performance**: Optimize for performance, especially for low-resource environments

Before submitting changes, please test thoroughly on different browsers and devices.

## Browser Support

The frontend is designed to work on modern browsers including:
- Chrome (latest 2 versions)
- Firefox (latest 2 versions)
- Safari (latest 2 versions)
- Edge (latest 2 versions)

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.