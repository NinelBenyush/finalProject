import SectionTitle from "./SectionTitle";
import OnePieChart from "./OnePieChart";
import Graph from "./Graph";

//section on the home page
function HowItWork() {
  return (
    <section className="py-20 bg-gray-50" id="howWork">
      <div className="container mx-auto px-6">
        <div className="text-center mb-10">
          <SectionTitle text="How it works" />
        </div>
        <div className="grid p md:grid-cols-2 items-center gap-16 ">
          <div className="text-center md:text-left">
            <OnePieChart />
          </div>
          <h3 className="text-xl font-semibold text-gray-800 mb-4"> From the file you are uploading, the data will be divided into groups of sequences, with each sequence comprising 3 months.</h3>
        </div>
        <div className="grid md:grid-cols-2 items-center  gap-16">
          <Graph />
          <div className="text-center md:text-left">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">The results will cover the upcoming months, and each month's prediction is based on the preceding 3 months.</h3>
          </div>
        </div>
      </div>
    </section>
  );
}

export default HowItWork;
