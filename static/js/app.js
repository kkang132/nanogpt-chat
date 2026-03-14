document.addEventListener('DOMContentLoaded', () => {
    const chatContainer = document.getElementById('chatContainer');
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const chatCount = document.getElementById('chatCount');
    const statusCount = document.getElementById('statusCount');
    const loading = document.getElementById('loading');

    function addMessage(content, isUser, chatId) {
        const row = document.createElement('div');
        row.className = `message-row ${isUser ? 'user' : 'assistant'}`;

        const avatar = document.createElement('div');
        avatar.className = `avatar ${isUser ? 'user-avatar' : 'assistant-avatar'}`;
        avatar.textContent = isUser ? 'U' : 'N';

        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.textContent = content;

        row.appendChild(avatar);
        row.appendChild(bubble);
        chatContainer.appendChild(row);

        if (!isUser && chatId) {
            const feedbackRow = document.createElement('div');
            feedbackRow.className = 'feedback-row';

            const thumbsUp = document.createElement('button');
            thumbsUp.className = 'feedback-btn';
            thumbsUp.textContent = '\u25B2';
            thumbsUp.title = 'Good response';

            const thumbsDown = document.createElement('button');
            thumbsDown.className = 'feedback-btn';
            thumbsDown.textContent = '\u25BC';
            thumbsDown.title = 'Bad response';

            thumbsUp.addEventListener('click', () => rateResponse(chatId, 1, thumbsUp, thumbsDown));
            thumbsDown.addEventListener('click', () => rateResponse(chatId, 0, thumbsUp, thumbsDown));

            feedbackRow.appendChild(thumbsUp);
            feedbackRow.appendChild(thumbsDown);
            chatContainer.appendChild(feedbackRow);
        }

        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    async function rateResponse(chatId, rating, upBtn, downBtn) {
        upBtn.disabled = true;
        downBtn.disabled = true;

        try {
            const response = await fetch('/rate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ chat_id: chatId, rating: rating }),
            });

            if (response.ok) {
                if (rating === 1) {
                    upBtn.classList.add('selected-up');
                } else {
                    downBtn.classList.add('selected-down');
                }
            } else {
                upBtn.disabled = false;
                downBtn.disabled = false;
            }
        } catch (error) {
            console.error('Rating error:', error);
            upBtn.disabled = false;
            downBtn.disabled = false;
        }
    }

    async function sendMessage() {
        const message = messageInput.value.trim();
        if (!message) return;

        addMessage(message, true);
        messageInput.value = '';
        sendBtn.disabled = true;
        loading.classList.add('show');
        chatContainer.scrollTop = chatContainer.scrollHeight;

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message }),
            });

            const data = await response.json();
            loading.classList.remove('show');

            if (data.response) {
                addMessage(data.response, false, data.chat_id);
                chatCount.textContent = data.chat_count;
                statusCount.textContent = data.chat_count + ' conversations';
            } else {
                addMessage('Error generating response.', false);
            }
        } catch (error) {
            loading.classList.remove('show');
            addMessage('Connection error. Is the server running?', false);
            console.error('Error:', error);
        } finally {
            sendBtn.disabled = false;
            messageInput.focus();
        }
    }

    sendBtn.addEventListener('click', sendMessage);

    document.addEventListener('keydown', (e) => {
        if (document.activeElement === messageInput && !e.metaKey && !e.ctrlKey && !e.altKey) {
            e.stopPropagation();
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        }
    }, true);

    // Load initial stats
    fetch('/stats')
        .then(r => r.json())
        .then(data => {
            chatCount.textContent = data.chat_count;
            statusCount.textContent = data.chat_count + ' conversations';
        });

    messageInput.focus();
});
