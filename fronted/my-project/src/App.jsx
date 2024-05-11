import Navbar from "./components/Navbar";
import Hero from "./components/Hero";
import Start from "./components/Start";
import About from "./components/About";
import React, { useEffect } from 'react';


function App(){

  useEffect(() => {
    fetch('http://localhost:5000/')
      .then(response => response.json())
      .then(data => {
        console.log('Data from Flask API:');
        // You can also handle the data further, e.g., update state
      })
      .catch(error => {
        console.error('Error:', error);
      });
  }, []);


  return (
    <>
    <Navbar />
    <Hero />
    <Start/>
    <About />
    </>
  )
}

export default App;