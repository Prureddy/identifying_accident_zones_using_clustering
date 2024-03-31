// scripts.js
document.addEventListener('DOMContentLoaded', function() {
    var form = document.querySelector('form');

    form.addEventListener('submit', function(event) {
        event.preventDefault();

        var algorithm = document.getElementById('algorithm').value;
        var formData = new FormData();
        formData.append('algorithm', algorithm);

        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => response.text())
        .then(data => {
            document.getElementById('results').innerHTML = data;
        })
        .catch(error => console.error('Error:', error));
    });
});
