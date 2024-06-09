import ProfileNavbar from "./ProfileNavbar";
import Footer from "./Footer";
import React, { useEffect, useState } from 'react';
import axios from "axios";

function UploadedFiles(){
    const [ file, setFiles] = useState([]);

    useEffect(() =>{
        async function fetchFiles() {
            try {
              const response = await axios.get('http://localhost:5000/profile/files');
              setFiles(response.data.files);
            } catch (error) {
              console.error('Error fetching files', error);
            }
          }
      
          fetchFiles();
    }, []);

    return (
        <>
        <ProfileNavbar />
        <div className="overflow-x-auto h-screen m-10">
        <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-md">
  <table className="table">

    <thead>
      <tr>
        <th>#</th>
        <th>File name</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody><tr>
        <th>1</th>
        <td>Cy Ganderton</td>
        <td>Quality Control Specialist</td>
      </tr>
      {/* row 2 */}
      <tr>
        <th>2</th>
        <td>Hart Hagerty</td>
        <td>Desktop Support Technician</td>
      </tr>
    </tbody>
  </table>
  </div>
</div>
<Footer />
</>
    )
}

export default UploadedFiles;