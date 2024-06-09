import React from "react";
import ProfileNavbar from "./ProfileNavbar";
import Footer from "./Footer";
import profileImg from "../assets/profileImg.png"
import Sidebar from "./Sidebar";


function ProfilePage() {
  return (
    <>
      <ProfileNavbar transparent />
      <Sidebar />
      <Footer />
    </>
  );
}

export default ProfilePage;