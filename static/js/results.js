document.addEventListener('DOMContentLoaded', function() {
    // console.log(1);
    fetch('/static/results.json')
        .then(response => response.json())
        .then(sentiments => {
            // Initialize the pie chart with fetched sentiment data
            var ctx = document.getElementById('redditChart').getContext('2d');
            console.log(sentiments);
            var sentimentPieChart = new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: ['Positive', 'Neutral', 'Negative'],
                    datasets: [{
                        label: 'Sentiment Analysis',
                        data: sentiments,
                        backgroundColor: [
                            'rgba(75, 192, 192, 0.2)', // Positive: Teal
                            'rgba(54, 162, 235, 0.2)', // Neutral: Blue
                            'rgba(255, 99, 132, 0.2)'  // Negative: Red
                        ],
                        borderColor: [
                            'rgba(75, 192, 192, 1)',
                            'rgba(54, 162, 235, 1)',
                            'rgba(255, 99, 132, 1)'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'top',
                        }
                    }
                }
            });
        })
        .catch(error => console.error('Error fetching sentiment data: ', error));
});


document.addEventListener('DOMContentLoaded', function() {
    // console.log(2);
    fetch('/static/trends.json')
        .then(response => response.json())
        .then(data => {
            var ctx=document.getElementById('search-interest')
            ctx.textContent= `${data}`;
        }
    )
    .catch(error => console.error('Error fetching trend data: ', error));
}
);

document.addEventListener('DOMContentLoaded', function() {
    fetch('/news-data')
        .then(response => response.json())
        .then(sentiments => {
            // Initialize the pie chart with fetched sentiment data
            var ctx = document.getElementById('newsChart').getContext('2d');
            var sentimentPieChart = new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: ['Positive','Negative', 'Neutral' ],
                    datasets: [{
                        label: 'Sentiment Analysis',
                        data: sentiments,
                        backgroundColor: [
                            'rgba(75, 192, 192, 0.2)', // Positive: Teal
                            'rgba(54, 162, 235, 0.2)', // Neutral: Blue
                            'rgba(255, 99, 132, 0.2)'  // Negative: Red
                        ],
                        borderColor: [
                            'rgba(75, 192, 192, 1)',
                            'rgba(54, 162, 235, 1)',
                            'rgba(255, 99, 132, 1)'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'top',
                        }
                    }
                }
            });
        })
        .catch(error => console.error('Error fetching sentiment data: ', error));
});

