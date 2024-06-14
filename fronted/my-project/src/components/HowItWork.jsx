import SectionTitle from "./SectionTitle";
import OnePieChart from "./OnePieChart";

function HowItWork(){
    return (
        <>
         <section className="py-20 align-element bg-gray-50" id="howWork">
            <div className="align-element grid md:grid-cols-2 items-center gap-16">
                <SectionTitle text="How it work"/>
            </div>
            <article className=" align-element grid md:grid-cols-2 items-center gap-16">
                    <OnePieChart/>
                </article>
            <div>
                <p className="py-6 align-element grid md:grid-cols-2 items-center gap-16">
                From the file you are uploading, the data will be divided into groups of sequences, with each sequence comprising 3 months.
                </p>

            </div>

        </section>
        </>
    )
}

export default HowItWork;