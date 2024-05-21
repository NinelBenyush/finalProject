import SectionTitle from "./SectionTitle";
import StartCard from "./StartCard";
import {start} from "../data";
import React, { useState } from 'react';
import axios from "axios";


function Start(){

   

    return (
        <section className="py-20 align-element" id="start">
           <SectionTitle text='Lets Start' /> 
           <div className="py-16 grid gap-8 md:grid-cols-2 lg:grid-cols-3">
            {start.map((s) =>{
                return <StartCard key={s.id} {...s} />;
            })}
            </div> 
        </section>
    )
}

export default Start;