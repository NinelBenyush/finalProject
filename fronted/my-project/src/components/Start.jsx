import React, { useState, useEffect } from 'react';
import SectionTitle from "./SectionTitle";
import StartCard from "./StartCard";
import { start } from "../data";
import axios from "axios";

function Start() {
  const [file, setFile] = useState(null);

  useEffect(() => {
    const fileInput = document.getElementById('fileInput');
    const submitButton = document.getElementById('submitButton');

    if (fileInput) {
      fileInput.addEventListener('change', handleFileChange);
    }

    if (submitButton) {
      submitButton.addEventListener('click', handleFileUpload);
    }

    return () => {
      if (fileInput) {
        fileInput.removeEventListener('change', handleFileChange);
      }

      if (submitButton) {
        submitButton.removeEventListener('click', handleFileUpload);
      }
    };
  }, []);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleFileUpload = (event) => {
    event.preventDefault();
    if (file) {
      const formData = new FormData();
      formData.append('file', file);

      axios.post('http://localhost:5000', formData)
        .then(response => {
          console.log('File uploaded successfully:', response.data);
        })
        .catch(error => {
          console.error('Error uploading file:', error);
        });
    }
    console.log("submit got clicked");
  };
  useEffect(() => {
    const submitButton = document.getElementById('submitButton');

    if (submitButton) {
      submitButton.addEventListener('click', handleFileUpload);
    }

    return () => {
      if (submitButton) {
        submitButton.removeEventListener('click', handleFileUpload);
      }
    };
  }, [handleFileUpload]);

  return (
    <section className="py-20 align-element bg-gray-50" id="start">
      <SectionTitle text="Let's Start" />
      <div className="py-16 grid gap-8 md:grid-cols-2 lg:grid-cols-3">
        {start.map((s) => (
          <div key={s.id} className="h-full flex">
            <StartCard {...s} />
          </div>
        ))}
      </div>
    </section>
  );
}

export default Start;
