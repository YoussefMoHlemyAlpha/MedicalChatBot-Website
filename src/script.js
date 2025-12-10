const messagesDiv = document.getElementById("messages");
const questionInput = document.getElementById("question");

// Shopping cart state
let currentUser = JSON.parse(localStorage.getItem('medical_current_user')) || null;
let cart = [];

function getCartKey() {
    return currentUser ? `medical_cart_${currentUser.username}` : 'medical_cart_guest';
}

function loadCart() {
    const key = getCartKey();
    cart = JSON.parse(localStorage.getItem(key)) || [];
    updateCartDisplay();
}

// Initial load
// Initial load moved to DOMContentLoaded
// Removed call to updateCartDisplay here to ensure DOM is ready first

function updateAuthUI() {
    const authLink = document.getElementById('auth-link');
    if (currentUser) {
        authLink.innerHTML = `<a href="#" onclick="logoutUser()">Welcome, ${currentUser.username} (Logout)</a>`;
    } else {
        authLink.innerHTML = `<a href="#" onclick="showAuthModal()">Login / Register</a>`;
    }
}

window.showAuthModal = function () {
    // Create modal wrapper
    const modal = document.createElement('div');
    modal.id = 'auth-modal'; // Specific ID for styling
    modal.className = 'modal-overlay'; // Re-use an overlay class if we make one, or style ID

    // Create content container
    const content = document.createElement('div');
    content.className = 'auth-content';

    // Default to Login view
    content.innerHTML = `
        <h2 id="auth-title">Login</h2>
        <div class="form-group">
            <label>Username</label>
            <input type="text" id="auth-username" placeholder="Username">
        </div>
        <div class="form-group">
            <label>Password</label>
            <input type="password" id="auth-password" placeholder="Password">
        </div>
        
        <!-- Extra field for register only, hidden by default -->
        <div class="form-group" id="phone-group" style="display:none;">
            <label>Phone (for orders)</label>
            <input type="tel" id="auth-phone" placeholder="Phone Number">
        </div>

        <div class="form-actions">
            <button class="btn-confirm" id="auth-action-btn" onclick="loginUser()">Login</button>
            <button class="btn-cancel" onclick="closeAuthModal()">Cancel</button>
        </div>
        <p style="margin-top:15px; font-size:0.9em;">
            <span id="auth-toggle-text">Don't have an account?</span> 
            <a href="#" onclick="toggleAuthMode()" style="color:#27ae60; font-weight:bold;">Click here</a>
        </p>
    `;

    modal.appendChild(content);
    document.body.appendChild(modal);
};

window.closeAuthModal = function () {
    const modal = document.getElementById('auth-modal');
    if (modal) modal.remove();
};

let isRegisterMode = false;
window.toggleAuthMode = function () {
    isRegisterMode = !isRegisterMode;
    const title = document.getElementById('auth-title');
    const btn = document.getElementById('auth-action-btn');
    const toggleText = document.getElementById('auth-toggle-text');
    const phoneGroup = document.getElementById('phone-group');

    if (isRegisterMode) {
        title.textContent = 'Register';
        btn.textContent = 'Register';
        btn.onclick = registerUser;
        toggleText.textContent = 'Already have an account?';
        phoneGroup.style.display = 'block';
    } else {
        title.textContent = 'Login';
        btn.textContent = 'Login';
        btn.onclick = loginUser;
        toggleText.textContent = "Don't have an account?";
        phoneGroup.style.display = 'none';
    }
};

window.registerUser = function () {
    const username = document.getElementById('auth-username').value.trim();
    const password = document.getElementById('auth-password').value.trim();
    const phone = document.getElementById('auth-phone').value.trim();

    if (!username || !password || !phone) {
        alert("Please fill in all fields.");
        return;
    }

    const users = JSON.parse(localStorage.getItem('medical_users')) || [];

    // Check if user exists
    if (users.find(u => u.username === username)) {
        alert("Username already taken.");
        return;
    }

    const newUser = { username, password, phone };
    users.push(newUser);
    localStorage.setItem('medical_users', JSON.stringify(users));

    // Auto login
    currentUser = newUser;
    localStorage.setItem('medical_current_user', JSON.stringify(currentUser));

    updateAuthUI();
    closeAuthModal();
    showNotification(`Welcome, ${username}!`, 'success');
};

window.loginUser = function () {
    const usernameInput = document.getElementById('auth-username');
    if (!usernameInput) return;

    const username = usernameInput.value.trim();
    const password = document.getElementById('auth-password').value.trim();

    if (!username || !password) {
        alert("Please enter username and password.");
        return;
    }

    const users = JSON.parse(localStorage.getItem('medical_users')) || [];
    const user = users.find(u => u.username === username && u.password === password);

    if (user) {
        currentUser = user;
        localStorage.setItem('medical_current_user', JSON.stringify(currentUser));
        loadCart(); // Switch to user's cart
        updateAuthUI();
        closeAuthModal();
        showNotification(`Welcome back, ${user.username}!`, 'success');
    } else {
        alert("Invalid username or password.");
    }
};

window.logoutUser = function () {
    currentUser = null;
    localStorage.removeItem('medical_current_user');
    loadCart(); // Switch back to guest cart
    updateAuthUI();
    showNotification("Logged out successfully", 'success');
};

function addMessage(sender, text, type) {
    let className = 'bot-message';
    if (type === 'user') className = 'user-message';
    else if (type === 'commerce') className = 'commerce-message';

    const p = document.createElement('p');
    p.className = className;
    p.innerHTML = `<strong>${sender}:</strong> ${text}`;
    messagesDiv.appendChild(p);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

async function ask() {
    console.log("ask() called");
    const question = questionInput.value.trim();
    if (!question) {
        alert("Please enter a question!");
        return;
    }

    addMessage("You", question, "user");
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

        // Remove loading message
        messagesDiv.removeChild(loadingMsg);

        if (data.intent) {
            // Commerce response from BERT
            addMessage("Smart Agent", data.answer, "commerce");
        } else if (data.answer) {
            addMessage("Medical Bot", data.answer, "bot");
        } else if (data.error) {
            addMessage("Bot", "Error: " + data.error, "bot");
        } else {
            addMessage("Bot", "Sorry, no answer received.", "bot");
        }
    } catch (error) {
        messagesDiv.removeChild(loadingMsg);
        console.error("Error:", error);
        addMessage("Bot", "Error connecting to server.", "bot");
    }
}

// Shopping Cart Functions - Exposed globally for onclick handlers
window.addToCart = function (medicineName, price) {
    cart.push({
        name: medicineName,
        price: price,
        quantity: 1,
        timestamp: new Date()
    });

    saveCart();
    showNotification(` ${medicineName} ($${price}) added to cart!`, 'success');
    updateCartDisplay();
};

window.buyNow = function (medicineName, price) {
    cart.push({
        name: medicineName,
        price: price,
        quantity: 1,
        timestamp: new Date()
    });
    saveCart();
    updateCartDisplay();

    showCart(); // Open modal
    checkout(); // Switch to checkout form
};

function updateCartDisplay() {
    const cartCount = cart.length;
    let cartBtn = document.getElementById('cart-btn');

    if (!cartBtn) {
        // Create cart button if it doesn't exist
        cartBtn = document.createElement('button');
        cartBtn.id = 'cart-btn';
        cartBtn.className = 'cart-button';
        cartBtn.onclick = showCart;
        document.body.appendChild(cartBtn);
    }

    cartBtn.innerHTML = `🛒 Cart (${cartCount})`;
}

window.showCart = function () {
    // Create modal wrapper
    const modal = document.createElement('div');
    modal.id = 'cart-modal';

    // Create content container
    const content = document.createElement('div');
    content.className = 'cart-content';

    // Add title
    const title = document.createElement('h2');
    title.textContent = '🛒 Your Shopping Cart';
    content.appendChild(title);

    // Add items list
    const itemsList = document.createElement('ul');
    itemsList.className = 'cart-items';

    let totalCost = 0;

    if (cart.length === 0) {
        itemsList.innerHTML = '<p style="padding:10px; text-align:center;">Your cart is empty.</p>';
    } else {
        cart.forEach((item, index) => {
            const itemTotal = item.price * item.quantity;
            totalCost += itemTotal;
            const li = document.createElement('li');
            li.innerHTML = `${item.name} - <strong>$${item.price}</strong> <button onclick="removeFromCart(${index})">Remove</button>`;
            itemsList.appendChild(li);
        });
    }
    content.appendChild(itemsList);

    // Show Total
    if (cart.length > 0) {
        const totalDiv = document.createElement('div');
        totalDiv.style.textAlign = 'right';
        totalDiv.style.fontWeight = 'bold';
        totalDiv.style.fontSize = '1.2em';
        totalDiv.style.margin = '10px 0';
        totalDiv.innerHTML = `Total: $${totalCost.toFixed(2)}`;
        content.appendChild(totalDiv);
    }

    // Add buttons container
    const btnContainer = document.createElement('div');
    btnContainer.style.marginTop = '15px';
    btnContainer.style.display = 'flex';
    btnContainer.style.gap = '10px';
    btnContainer.style.justifyContent = 'flex-end';

    // Add checkout button
    if (cart.length > 0) {
        const checkoutBtn = document.createElement('button');
        checkoutBtn.textContent = `Checkout ($${totalCost.toFixed(2)})`;
        checkoutBtn.onclick = checkout;
        btnContainer.appendChild(checkoutBtn);
    }

    // Add close button
    const closeBtn = document.createElement('button');
    closeBtn.textContent = 'Close';
    closeBtn.onclick = closeCart;
    btnContainer.appendChild(closeBtn);

    content.appendChild(btnContainer);

    // Assemble modal
    modal.appendChild(content);
    document.body.appendChild(modal);
};



window.closeCart = function () {
    const modal = document.getElementById('cart-modal');
    if (modal) modal.remove();
};

window.removeFromCart = function (index) {
    cart.splice(index, 1);
    saveCart();
    updateCartDisplay();
    closeCart();
    if (cart.length > 0) showCart();
};

window.checkout = function () {
    const modal = document.getElementById('cart-modal');
    if (!modal) return;

    // Clear content to show form
    const content = modal.querySelector('.cart-content');
    content.innerHTML = '';

    const title = document.createElement('h2');
    title.textContent = 'Checkout Details';
    content.appendChild(title);

    const userPhone = currentUser ? currentUser.phone : '';

    const form = document.createElement('div');
    form.className = 'checkout-form';
    form.innerHTML = `
        <div class="form-group">
            <label for="address">Delivery Address:</label>
            <textarea id="checkout-address" placeholder="Enter your full address..." rows="3"></textarea>
        </div>
        <div class="form-group">
            <label for="phone">Phone Number:</label>
            <input type="tel" id="checkout-phone" placeholder="e.g., 01234567890" value="${userPhone}">
        </div>
        <div class="form-actions">
            <button class="btn-cancel" onclick="showCart()">Back to Cart</button>
            <button class="btn-confirm" onclick="submitOrder()">Confirm Order</button>
        </div>
    `;
    content.appendChild(form);
};

window.submitOrder = function () {
    const address = document.getElementById('checkout-address').value.trim();
    const phone = document.getElementById('checkout-phone').value.trim();

    if (!address || !phone) {
        alert("Please provide both address and phone number.");
        return;
    }

    const total = cart.reduce((sum, item) => sum + (item.price * item.quantity), 0);

    // Show success notification
    showNotification(`Order placed! Details sent to ${phone}. Total: $${total.toFixed(2)}`, 'success');

    cart = [];
    saveCart();
    updateCartDisplay();
    closeCart();
};

function saveCart() {
    localStorage.setItem(getCartKey(), JSON.stringify(cart));
}

function showNotification(message, type) {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.classList.add('show');
    }, 10);

    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 2000);
}

// Allow Enter key to submit
questionInput.addEventListener("keypress", function (event) {
    if (event.key === "Enter") {
        ask();
    }
});

// Initialize cart on load
document.addEventListener('DOMContentLoaded', () => {
    loadCart();
    updateAuthUI();
});

