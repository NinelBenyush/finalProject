import Navbar from "./Navbar";
import Hero from "./Hero";
import Start from "./Start";
import About from "./About";
import Footer from "./Footer";
import HowItWork from "./HowItWork";

//include all the components for the home page
function Home(){
    return (
        <>
        <Navbar />
        <Hero />
        <Start />
        <About />
        <HowItWork />
        <Footer/>
        </>
    )
}

export default Home;