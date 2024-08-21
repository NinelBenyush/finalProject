import heroImg from "../assets/hero.svg";
import {FaTwitter, FaFacebook} from "react-icons/fa";

//landing page
function Hero(){
    return (
       <div className="bg-emerald-100 py-24 ">
        <div className="align-element grid md:grid-cols-2 items-center gap-8"> 
        <article>
            <h1 className="text-7xl font-bold tracking-wider">
                OrderBoost
            </h1>
            <p className="mt-4 text-3xl text-slate-700 capitalize tracking-wide">
            Our goal is to predict the inventory quantities needed for the upcoming months.
            </p>
            <p className="mt-2 text-lg text-slate-700 capitalize tracking-wide">
            making your life as a shop owner easier
            </p>
            <div className="flex-gap-x-4 mt-4">
                <a href="#">
                    <FaFacebook className="h-8 w-8 text-slate-500 hover:text-black duration-300"></FaFacebook>
                </a>
                <a href="#">
                    <FaTwitter className="h-8 w-8 text-slate-500 hover:text-black duration-300"></FaTwitter>
                </a>
            </div>
        </article>
        <article className="hidden md:block">
            <img src={heroImg} className="h-80 lg:h-96"></img>
        </article>
        </div>
       </div>
    )
}

export default Hero;

