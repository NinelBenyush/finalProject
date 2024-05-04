import SectionTitle from "./SectionTitle";
import aboutImg from "../assets/about.svg";

function About(){
    return (
        <section className="bg-white py-20 " id="about">
            <div className="align-element grid md:grid-cols-2 items-center gap-16">
                <img src={aboutImg} className="w-full h-64"/>
                <article>
                    <SectionTitle text="About us"/>
                    <p className="text-slate-600 mt-8 leading-loose">
                       The entire store may face situations of either stock shortages or excess inventory, both of which can result in financial losses. To address this issue, we decided to develop OrderBoost. 
                    </p>
                    <p className="text-slate-600 mt-8 leading-loose">
                        The planning process relies on analyzing historical inventory data using machine learning tools to forecast the amount of inventory needed to order for the upcoming months.
                    </p>
                </article>
            </div>

        </section>

    )
}

export default About;