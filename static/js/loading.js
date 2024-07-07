function checkStatus() {
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'results') {
                window.location.href = '/results'; // Redirect to results page
            } else {
                setTimeout(checkStatus, 1000); // Check again in one second
            }
        })
        .catch(error => console.error('Error checking status:', error));
}

window.onload = checkStatus; // Start checking when page loads