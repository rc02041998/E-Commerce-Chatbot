### **E-Commerce Chatbot using BERT**

This project is a Streamlit-based e-commerce chatbot that answers user questions about products using a pretrained BERT model for Question Answering (QA). The chatbot works on the Flipkart Product Dataset, enabling users to ask queries like product price, category, and description.

---

### **Features**

- **Product Selection**: Users can select a product from the Flipkart dataset.
- **Question Answering**: Answers user queries using BERT for QA.
- **Contextual Understanding**: Leverages product-specific context for precise answers.
- **Regex-based Post-Processing**: Extracts specific details (e.g., price) from model responses.
- **Interactive UI**: Built with Streamlit for a smooth user experience.

---

### **Setup Instructions**

#### 1. **Clone the Repository**
```bash
git clone https://github.com/rc02041998/E-Commerce-Chatbot.git
cd E-Commerce-Chatbot
```

#### 2. **Install Dependencies**
Install the required Python packages:
```bash
pip install -r requirements.txt
```

#### 3. **Add the Dataset**
Ensure the Flipkart Product Dataset (`flipkart_product_dataset.csv`) is in the project directory.

#### 4. **Run the Application**
Start the Streamlit app:
```bash
streamlit run app.py
```

---

### **Usage**

1. Select a product from the sidebar.
2. View product details like name, category, price, and description.
3. Ask questions about the product, e.g.:
   - "What is the price of this product?"
   - "What is the category?"
   - "Describe the product."

4. View the chatbot's response in real-time.

---

### **Technologies Used**

- **Language Model**: Pretrained BERT (`bert-base-uncased`).
- **Frontend**: Streamlit for UI.
- **Backend**: PyTorch for running the model.
- **Dataset**: Flipkart Product Dataset.

---

### **Contributing**

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Feature description"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a Pull Request.

---

### **License**
This project is licensed under the MIT License.

---

Feel free to contribute to this project or suggest enhancements! 😊