import img from "../assets/insertData2.jpg";
import React, {useState} from "react";
import axios from "axios";
import Navbar from "./Navbar";
import Footer from "./Footer";


function InsertData(){

    const [fileName, setFileName] = useState('');
    const [ file, setFile] = useState(null);

    const handleFile = (event) => {
        const file = event.target.files[0];
        if (file){
            setFile(file)
            setFileName(file.name);
        }
        else {
            setFileName('');
        }
    }

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
        } catch (error) {
          console.error('Error uploading file:', error);
        }
      };

    return (
      <div>
        <Navbar />
<div className="flex justify-center items-center min-h-screen">
  <div className="card card-side bg-base-100 shadow-xl" style={{ width: '600px' }}>
    <figure>
      <img src={img} alt="Movie"  />
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

      <br></br>
      <p>Click the button down below to upload</p>
      <div className="card-actions justify-end">
        <button className="btn bg-emerald-100" onClick={handleUpload}>Upload</button>
      </div>
    </div>
  </div>
</div>
<Footer />
</div>


    )
}

export default InsertData;