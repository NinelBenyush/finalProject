import img from "../assets/insertData2.jpg";
import React, { useEffect, useState } from "react";
import axios from "axios";
import Footer from "./Footer";
import MiniNavbar from "./MiniNabvar";
import { useNavigate } from "react-router-dom";

function InsertData() {
  const navigate = useNavigate();
  const [fileName, setFileName] = useState('');
  const [file, setFile] = useState(null);
  const [successUpload, setSuccessUpload] = useState(false);

  const handleFile = (event) => {
    const file = event.target.files[0];
    if (file) {
      setFile(file);
      setFileName(file.name);
      setSuccessUpload(false); 
    } else {
      setFileName('');
    }
  };

  const handleUpload = async () => {
    if (!file) {
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5000/upload-file', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      console.log('File uploaded successfully:', response.data);
      // Reset the file state after successful upload
      setFile(null);
      setFileName('');
      setSuccessUpload(true); 
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  useEffect(() => {
    if(successUpload){
      const timer = setTimeout(()=>{
        setSuccessUpload(false);
        navigate("/profile");
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [successUpload]);

  return (
    <div>
      <MiniNavbar />
      {successUpload && (
              <div className="relative items-center w-full px-5 py-12 mx-auto md:px-12 lg:px-24 max-w-7xl">
                <div className="p-6 border-l-4 border-green-500 rounded-r-xl bg-green-50">
                  <div className="flex">
                    <div className="flex-shrink-0">
                      <svg className="w-5 h-5 text-green-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"></path>
                      </svg>
                    </div>
                    <div className="ml-3">
                      <div className="text-sm text-green-600">
                        <p>We got your file</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
      <div className="flex justify-center items-center min-h-screen">
        <div className="card card-side bg-base-100 shadow-xl" style={{ width: '600px' }}>
          <figure>
            <img src={img} alt="Insert Data" />
          </figure>
          <div className="card-body">
            <h2 className="card-title">Here you need to choose a file</h2>

            <div className="flex items-center max-w-xs">
              <label
                htmlFor="files"
                className="btn bg-emerald-500 hover:bg-emerald-600 text-white font-semibold py-1 px-3 rounded cursor-pointer text-sm"
              >
                Choose File
              </label>
              <input
                id="files"
                type="file"
                className="hidden"
                onChange={handleFile}
              />
              <span className="ml-2 text-gray-700 text-sm">{fileName || 'No file selected'}</span>
            </div>

            <br />
            <p>Click the button down below to upload</p>
            <div className="card-actions justify-end">
              <button className="btn bg-emerald-100" onClick={handleUpload}>Upload</button>
            </div>
          </div>
        </div>
      </div>
      
      <Footer />
    </div>
  );
}

export default InsertData;