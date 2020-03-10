(function ($) {

    try {
        $(document).ready(function(){
            $("#job_id").change(async function(){
                let jobID = $( ".selectable:selected").text()
                if (jobID.length > 0) {
                    if (jobID != "Create New Job") {
                        let response = await fetch('/jobs/' + jobID + '/description')
                        let description = await response.text()
                        // console.log(jobID)
                        $("#job_description").text(description);
                        $("#job_description").attr({"disabled": "disabled"});
                        $("#newJobDiv").attr({"hidden": "hidden"});
                        $("#candidateNum").attr({"hidden": null});
                        $("#submit").attr({"hidden": null});
                        $("#submitCreate").attr({"hidden": "hidden"});
                    }
                    else {
                        $("#job_description").text("");
                        $("#job_description").attr({"disabled": null});
                        $("#candidateNum").attr({"hidden": "hidden"});
                        $("#submit").attr({"hidden": "hidden"});
                        $("#submitCreate").attr({"hidden": null});
                        $("#newJobDiv").attr({"hidden": null});
                    }
                }
                else {
                    $("#job_description").text("");
                    $("#newJobDiv").attr({"hidden": "hidden"});
                    $("#candidateNum").attr({"hidden": "hidden"});
                    $("#submit").attr({"hidden": "hidden"});
                    $("#submitCreate").attr({"hidden": null});
                }
            });

            $("#submit").click(async function(){

                let jobID = $( "select option:selected" ).text()
                let numApp = $("#num_app").val()
                let allCandidates = $("#allCandidates").val()
                document.location.href = '/jobs/' + jobID + '/candidates?numApp=' + numApp + '&allCandidates=' + allCandidates

            });

            $("#submitCreate").click(async function(){
                let jobID = $("#newJobId").val()
                let jobDescription = $("#job_description").val()
                let body = {
                        "jobID": jobID,
                        "jobDescription": jobDescription
                    }

                await fetch('/jobs/', {
                    "headers": {
                        "Content-Type": "application/JSON"
                    },
                    "method": "POST",
                    "body": JSON.stringify(body)
                })
                document.location.href = '/'
            });

        });

    } catch (err) {
        console.log(err);
    }


})(jQuery);