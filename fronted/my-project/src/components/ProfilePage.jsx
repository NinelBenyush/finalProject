import React, { useEffect, useState } from "react";
import ProfileNavbar from "./ProfileNavbar";
import Footer from "./Footer";
import profileImg from "../assets/profileImg.png";
import Sidebar from "./Sidebar";

function ProfilePage() {
  const [showResAlert, setShowResAlert] = useState(false);
  const [message, setMessage] = useState("");

  useEffect(() => {
    fetch("http://localhost:5000/get-res")
      .then((response) => response.json())
      .then((data) => {
        setMessage(data.message);
        setShowResAlert(true);
      })
      .catch((error) => console.error("Error:", error));
  }, []);

  return (
    <>
      <ProfileNavbar transparent />
      <Sidebar />
      <Footer />
      {showResAlert && (
        <div className="fixed top-0 left-0 w-full h-full flex justify-center items-center bg-gray-800 bg-opacity-50">
          <div className="bg-white p-8 rounded-lg shadow-md">
            <h2 className="text-xl font-bold mb-4">We just want to say</h2>
            <p>{message}</p>
            <button
              className="mt-4 bg-emerald-500 hover:bg-emerald-700 text-white font-bold py-2 px-4 rounded"
              onClick={() => setShowResAlert(false)}
            >
              Close
            </button>
          </div>
        </div>
      )}
    </>
  );
}

export default ProfilePage;
