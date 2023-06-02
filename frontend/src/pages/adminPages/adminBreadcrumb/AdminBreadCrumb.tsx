import FiberManualRecordIcon from '@mui/icons-material/FiberManualRecord'
import { Breadcrumbs } from '@mui/material'
import { Typography } from '@mui/material'
import { Link, useLocation } from 'react-router-dom'

type Props = {
  breadcrumbs?: JSX.Element[]
  breadcrumbTitle: string
}

const AdminBreadCrumb = (props: Props) => {
  const { breadcrumbTitle } = props
  const location = useLocation()
  const path = location.pathname
  const segments = path.split('/')

  const breadcrumbs = [
    <Link to={'/admin/dashboard'} key='1' className='breadcrumb__link'>
      Dashboard
    </Link>
  ]

  const removeSegments = segments.slice(2)
  removeSegments.forEach((segment, index) => {
    if (removeSegments.includes('profile') && segment === 'doctors') {
      breadcrumbs.push(
        <Link to={'/admin/doctors/list'} key='1' className='breadcrumb__link'>
          Doctors
        </Link>
      )
      return
    }
    if (removeSegments.includes('profile') && segment === 'profile') {
      breadcrumbs.push(
        <Typography key={index + 2} color='text.primary'>
          {segment.charAt(0).toUpperCase() + segment.slice(1)}
        </Typography>
      )
      return
    }
    breadcrumbs.push(
      <Typography key={index + 2} color='text.primary'>
        {segment.charAt(0).toUpperCase() + segment.slice(1)}
      </Typography>
    )
  })

  return (
    <div>
      <p className='admin__p--title'>{breadcrumbTitle}</p>
      <div className='admin__breadcrumds'>
        <Breadcrumbs
          separator={<FiberManualRecordIcon fontSize='small' />}
          aria-label='breadcrumb'
        >
          {breadcrumbs}
        </Breadcrumbs>
      </div>
    </div>
  )
}

export default AdminBreadCrumb
