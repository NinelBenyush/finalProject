import img from "../assets/insertData2.jpg";
import React, {useState} from "react";

function InsertData(){

    const [fileName, setFileName] = useState('');

    const handleFile = (event) => {
        const file = event.target.files[0];
        if (file){
            setFileName(file.name);
        }
        else {
            setFileName('');
        }
    }

    return (
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
        <button className="btn bg-emerald-100">Upload</button>
      </div>
    </div>
  </div>
</div>


    )
}

export default InsertData;