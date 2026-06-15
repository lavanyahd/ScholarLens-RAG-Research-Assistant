import axios from "axios";

const API_BASE_URL = "https://scholarlens-rag-research-assistant.onrender.com";
export const uploadPdf = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await axios.post(`${API_BASE_URL}/upload-pdf`, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });

  return response.data;
};

export const askQuestion = async (paperId, question, advancedMode = false) => {
  const response = await axios.post(`${API_BASE_URL}/ask`, {
    paper_id: paperId,
    question: question,
    advanced_mode: advancedMode,
  });

  return response.data;
};

export const getSummary = async (paperId) => {
  const response = await axios.post(`${API_BASE_URL}/summary`, {
    paper_id: paperId,
  });

  return response.data;
};