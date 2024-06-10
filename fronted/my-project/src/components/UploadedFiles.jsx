import ProfileNavbar from "./ProfileNavbar";
import Footer from "./Footer";
import React, { useEffect, useState } from 'react';
import axios from "axios";

function UploadedFiles() {
    const [files, setFiles] = useState([]);

    useEffect(() => {
        async function fetchFiles() {
            try {
                const response = await axios.get('http://localhost:5000/profile/files');
                if (response.data.status === 'success') {
                    setFiles(response.data.files);
                } else {
                    console.error('Failed to fetch files', response.data.message);
                }
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
                        <tbody>
                            {files.map((file, index) => (
                                <tr key={index}>
                                    <th>{index + 1}</th>
                                    <td>{file.filename}</td>
                                    <td>{file.description}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
            <Footer />
        </>
    );
}

export default UploadedFiles;
