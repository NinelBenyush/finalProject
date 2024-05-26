import Navbar from "./components/Navbar";
import Hero from "./components/Hero";
import Start from "./components/Start";
import About from "./components/About";
import React, { useEffect } from 'react';
import Login from "./components/Login";
import InsertData from "./components/InsertData";
import Register from "./components/Register";
import Footer from "./components/Footer";
import DetailsAccount from "./components/DetailsAccount";
import DownloadResult from "./components/DownloadResult";

function App(){
  

  useEffect(() => {
    fetch('http://localhost:5000/')
      .then(response => response.json())
      .then(data => {
        console.log('Data from  API:' + data.message);
      })
      .catch(error => {
        console.error('Error:', error);
      });
  }, []);


  return (
    <>
    <Navbar  />
    <Hero />
    <Start/>
    <About />
    <Login />
    <InsertData />
    <Register />
    <DetailsAccount />
    <DownloadResult />
    <Footer />
    </>
  )
}

export default App;