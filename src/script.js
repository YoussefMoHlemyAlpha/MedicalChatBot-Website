const questionInput = document.getElementById("question");
const messagesDiv = document.getElementById("messages");

function addMessage(sender, text) {
    const msg = document.createElement("p");

    if (sender === "You") {
        msg.className = "user-message"; 
    } else {
        msg.className = "bot-message";   
    }

    msg.innerHTML = `<strong>${sender}:</strong> ${text}`;
    messagesDiv.appendChild(msg);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;   
}


async function ask() {
    console.log("ask() called"); 
    const question = questionInput.value.trim();
    if (!question) {
        alert("Please enter a question!");
        return;
    }

    addMessage("You", question);
    questionInput.value = "";
    // Show loading message
    const loadingMsg = document.createElement("p");
    loadingMsg.className = "loading-message";
    loadingMsg.innerText = "Bot is typing...";
    messagesDiv.appendChild(loadingMsg);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    try {
        const response = await fetch("http://127.0.0.1:8000/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question })
        });

        console.log("Response status:", response.status); 
        const data = await response.json();
        console.log("Response data:", data); 
        if (data.answer) {
            addMessage("Bot", data.answer);
        } else {
            addMessage("Bot", "Sorry, no answer received.");
        }
    } catch (error) {
        console.error("Error:", error);
        addMessage("Bot", "Error connecting to server.");
    }
}

// Allow pressing Enter
questionInput.addEventListener("keypress", function (e) {
    if (e.key === "Enter") {
        ask();
    }
});
