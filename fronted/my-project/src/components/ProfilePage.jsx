import React, { useEffect, useState } from "react";
import ProfileNavbar from "./ProfileNavbar";
import Footer from "./Footer";
import Sidebar from "./Sidebar";
import axios from "axios";

function ProfilePage() {
  const [showResAlert, setShowResAlert] = useState(false);
  const [message, setMessage] = useState("");

  useEffect(() => {
    async function fetchRes() {
      try {
        const response = await axios.get("http://localhost:5000/get-res");
        const m = response.data.message;
        const lastAlertTime = localStorage.getItem("lastAlertTime");
        const currentTime = new Date().getTime();

        if (
          m === "You got the results, check in the results section" &&
          (!lastAlertTime || currentTime - parseInt(lastAlertTime) > 60000)
        ) {
          setMessage(m);
          setShowResAlert(true);
          localStorage.setItem("lastAlertTime", currentTime.toString());
        }
      } catch (error) {
        console.error("error ", error);
      }
    }
    fetchRes();
  }, []);

  const handleCloseAlert = () => {
    setShowResAlert(false);
  };

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
            onClick={handleCloseAlert}
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
