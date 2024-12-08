document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const textInput = document.getElementById('textInput');
    const fileInput = document.getElementById('fileInput');
    const generateButton = document.getElementById('generateAnswerButton');
    const responseText = document.getElementById('responseText');

    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const formData = new FormData();
            const file = fileInput?.files[0];
            const text = textInput?.value.trim();

            if (!file && !text) {
                alert('Please provide a file or enter text.');
                return;
            }

            if (file) formData.append('file', file);
            if (text) formData.append('text', text);

            if (generateButton) generateButton.disabled = true;
            if (responseText) responseText.textContent = 'Processing request...';

            try {
                const response = await fetch('http://localhost:6065/index/document', {
                    method: 'POST',
                    body: formData,
                    mode: 'no-cors'
                });

                if (responseText) {
                    responseText.textContent = 'Document processed successfully!';
                }

            } catch (error) {
                console.error('Error details:', error);
                
                if (responseText) {
                    responseText.textContent = `Error: ${error.message}`;
                }
                alert('Error occurred while processing the request');

            } finally {
                if (generateButton) generateButton.disabled = false;
            }
        });
    }
});