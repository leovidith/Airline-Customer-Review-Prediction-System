# **Airline Customer Review Prediction System**

## **Overview**
This project is designed to predict whether a customer review for an airline is positive or negative based on textual analysis. Leveraging BERT (Bidirectional Encoder Representations from Transformers), a state-of-the-art NLP model, the system effectively classifies reviews, assisting in gauging customer satisfaction and identifying areas for improvement. By transforming unstructured review data into actionable insights, this project showcases the power of machine learning in real-time sentiment analysis.

### **Key Phases:**
1. **Data Preprocessing**:
   - Cleaned the dataset by handling missing values and ensuring data integrity.
   - Transformed customer ratings into binary sentiment labels (positive/negative).
   - Preprocessed review texts by removing irrelevant characters, URLs, and special symbols, standardizing text for consistency.

2. **Text Encoding**:
   - Utilized the BERT tokenizer (`bert-base-uncased`) to convert raw text into model-compatible token sequences.
   - Ensured uniform input lengths by applying padding and truncation strategies.

3. **Model Training Pipeline**:
   - Constructed a TensorFlow data pipeline to enhance training efficiency by batching and shuffling the dataset.
   - Fine-tuned BERT for binary sentiment classification, optimizing model parameters for superior predictive performance.

4. **Model Architecture**:
   - Employed BERT's transformer layers to extract rich feature representations of the text, followed by custom dense layers for final classification.
   - Applied dropout and batch normalization layers to improve model robustness and prevent overfitting.

5. **Training and Validation Strategy**:
   - Split the dataset into training (80%) and validation (20%) subsets to ensure a balanced evaluation.
   - Incorporated early stopping and learning rate reduction to optimize training and prevent overfitting.

## **Results**
The model achieved an outstanding **99.6% validation accuracy**, reflecting its ability to effectively distinguish between positive and negative reviews. The high performance highlights the effectiveness of fine-tuning BERT for sentiment analysis tasks, especially when working with well-prepared and cleaned data.

### **Performance Metrics**:
- **Validation Accuracy**: 99.6%
- **Training Accuracy**: Consistent high performance, demonstrating model stability and reliability.

## **Agile Features**

1. **Preprocessing Pipeline**:
   - Systematically cleaned and preprocessed data to ensure high-quality inputs for the model.
   - Standardized text by converting to lowercase and eliminating noise, making it suitable for NLP tasks.

2. **Model Design**:
   - Fine-tuned BERT for binary classification to predict customer sentiments effectively.
   - Incorporated batch normalization and dropout mechanisms to enhance generalization and model performance.

3. **Training Optimization**:
   - Applied dynamic learning rate adjustments and early stopping to ensure efficient training and mitigate overfitting.

4. **Model Evaluation**:
   - Analyzed the model's performance through accuracy and loss curves, ensuring continuous monitoring of training progress.

## **Conclusion**
This project demonstrates the transformative potential of advanced NLP techniques in understanding customer sentiment. With a **99.6% validation accuracy**, the model proves its capability to provide valuable insights into customer experiences. Its successful application could significantly enhance customer feedback analysis, helping businesses refine their services and customer support strategies. Future enhancements may include expanding the dataset, incorporating additional features, and testing other transformer models to further elevate performance.
